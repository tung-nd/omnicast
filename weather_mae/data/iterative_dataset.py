import os
import numpy as np
import torch
import h5py

from torch.utils.data import Dataset
from glob import glob
import threading
from concurrent.futures import ThreadPoolExecutor
from weather_mae.models.vaes.continuous_vae_transformer import DiagonalGaussianDistribution

def get_data_given_path(path, variables, stack_axis=0):
    """Read data using memory mapping"""
    # 'core' driver with backing_store=False keeps the file in memory
    # swmr=True enables Single Writer Multiple Reader mode if needed
    with h5py.File(path, 'r', driver='core', backing_store=False) as f:
        input_group = f['input']
        # Since the file is memory mapped, this doesn't actually load the data
        # until it's accessed during the np.stack operation
        arrays = [np.array(input_group[v]) for v in variables]
        # This is where the actual data loading happens
        return np.stack(arrays, axis=stack_axis)


class ERA5ImageDataset(Dataset):
    def __init__(
        self,
        root_dir,
        variables,
        transform,
        return_filename=False,
    ):
        super().__init__()
        
        self.root_dir = root_dir
        self.variables = variables
        self.transform = transform
        self.return_filename = return_filename
        
        file_paths = glob(os.path.join(root_dir, '*.h5'))
        self.file_paths = sorted(file_paths)
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        path = self.file_paths[index]
        inp = get_data_given_path(path, self.variables)
        inp = torch.from_numpy(inp)
        if not self.return_filename:
            return self.transform(inp)
        return self.transform(inp), os.path.basename(path)


class ERA5VideoDataset(Dataset):
    def __init__(
        self,
        root_dir,
        variables,
        transform,
        steps,
        interval=6,
        data_freq=6,
        num_workers=16
    ):
        super().__init__()
        
        self.root_dir = root_dir
        self.variables = variables
        self.transform = transform
        self.steps = steps
        self.interval = interval
        self.data_freq = data_freq
        
        file_paths = glob(os.path.join(root_dir, '*.h5'))
        file_paths = sorted(file_paths)
        self.inp_file_paths = file_paths[:-(steps*interval // data_freq)]
        self.file_paths = file_paths
        
        # Create a single thread pool for the dataset
        self.num_workers = num_workers
        self._executor = None
        self._executor_lock = threading.Lock()
    
    @property
    def executor(self):
        """Lazy initialization of thread pool to ensure it's created in the correct thread"""
        if self._executor is None:
            with self._executor_lock:
                if self._executor is None:  # Double-check under lock
                    self._executor = ThreadPoolExecutor(max_workers=self.num_workers)
        return self._executor
    
    def _load_and_transform_frame(self, path):
        """Load and transform a single frame"""
        frame = get_data_given_path(path, self.variables)
        frame = torch.from_numpy(frame)
        frame = self.transform(frame)
        return frame
    
    def __len__(self):
        return len(self.inp_file_paths)
    
    def __getitem__(self, index):
        # Get paths for all frames in the sequence
        sequence_paths = [
            self.file_paths[index + (step * self.interval) // self.data_freq]
            for step in range(self.steps)
        ]
        
        # Submit all frame loading tasks to thread pool
        futures = [
            self.executor.submit(self._load_and_transform_frame, path)
            for path in sequence_paths
        ]
        
        # Collect results in order
        frames = [future.result() for future in futures]
        
        return torch.stack(frames, dim=1)  # CxTxHxW
    
    def __del__(self):
        """Clean up thread pool"""
        if self._executor is not None:
            self._executor.shutdown()


class ERA5LatentVideoDataset(Dataset):
    def __init__(
        self,
        raw_root_dir,
        raw_variables,
        raw_transform,
        latent_root_dir,
        latent_transform,
        steps,
        cond_steps,
        predict_dynamic,
        dynamic_transforms,
        interval=6,
        data_freq=6,
        sample_latent=False,
        num_workers=16,
        return_raw=False,
        return_filename=False,
    ):
        super().__init__()
        
        self.raw_root_dir = raw_root_dir
        self.raw_variables = raw_variables
        self.raw_transform = raw_transform
        self.latent_root_dir = latent_root_dir
        self.latent_transform = latent_transform
        self.steps = steps
        self.cond_steps = cond_steps
        self.predict_dynamic = predict_dynamic
        self.dynamic_transforms = dynamic_transforms
        self.interval = interval
        self.data_freq = data_freq
        self.sample_latent = sample_latent
        self.return_raw = return_raw
        self.return_filename = return_filename
        
        raw_file_paths = glob(os.path.join(raw_root_dir, '*.h5'))
        raw_file_paths = sorted(raw_file_paths)
        self.raw_inp_file_paths = raw_file_paths[:-((steps-1)*interval // data_freq)]
        self.raw_file_paths = raw_file_paths
        
        latent_file_paths = glob(os.path.join(latent_root_dir, '*.h5'))
        latent_file_paths = sorted(latent_file_paths)
        self.latent_inp_file_paths = latent_file_paths[:-((steps-1)*interval // data_freq)]
        self.latent_file_paths = latent_file_paths
        
        # Create a single thread pool for the dataset
        self.num_workers = num_workers
        self._executor = None
        self._executor_lock = threading.Lock()
    
    @property
    def executor(self):
        """Lazy initialization of thread pool to ensure it's created in the correct thread"""
        if self._executor is None:
            with self._executor_lock:
                if self._executor is None:  # Double-check under lock
                    self._executor = ThreadPoolExecutor(max_workers=self.num_workers)
        return self._executor
    
    def _load_latent(self, path):
        with h5py.File(path, 'r', driver='core', backing_store=False) as f:
            data = np.array(f['data'])
        latent = torch.from_numpy(data)
        if latent.shape[0] != self.latent_transform.mean.shape[0]:
            assert latent.shape[0] == 2 * self.latent_transform.mean.shape[0]
            latent = latent.unsqueeze(0)
            posterior = DiagonalGaussianDistribution(latent)
            if self.sample_latent:
                latent = posterior.sample().squeeze(0)
            else:
                latent = posterior.mean.squeeze(0)
        return latent
    
    def _load_and_transform_frame(self, path):
        """Load and transform a single frame"""
        frame = get_data_given_path(path, self.raw_variables)
        frame = torch.from_numpy(frame)
        frame = self.raw_transform(frame)
        return frame
    
    def __len__(self):
        return len(self.latent_inp_file_paths)
    
    def _getitem(self, file_paths, index, type):
        # Get paths for all frames in the sequence
        sequence_paths = [
            file_paths[index + (step * self.interval) // self.data_freq]
            for step in range(self.steps)
        ]
        
        # Submit all frame loading tasks to thread pool
        load_func = self._load_latent if type == 'latent' else self._load_and_transform_frame
        futures = [
            self.executor.submit(load_func, path)
            for path in sequence_paths
        ]
        
        # Collect results in order
        frames = [future.result() for future in futures]
        
        # transform latent
        if type == 'latent':
            latent_frames = torch.stack([self.latent_transform(frame) for frame in frames])

            # cond and dynamic frames
            cond_steps = self.cond_steps
            if cond_steps < self.steps:
                cond_frames = latent_frames[:cond_steps]
                dynamic_frames = torch.stack(frames[cond_steps:]) - torch.stack(frames[(cond_steps-1):-1])
                dynamic_frames = torch.stack([
                    self.dynamic_transforms(dynamic_frames[i]) for i in range(dynamic_frames.shape[0])
                ])
                latent_dynamics = torch.cat([cond_frames, dynamic_frames])
            else:
                latent_dynamics = latent_frames
            
            return latent_frames, latent_dynamics
        else:
            return torch.stack(frames)  # TxCxHxW
    
    def __getitem__(self, index):
        latent_data, latent_dynamics = self._getitem(self.latent_file_paths, index, 'latent')
        raw_data = self._getitem(self.raw_file_paths, index, 'raw') if self.return_raw else None
        if not self.return_filename:
            return latent_data, latent_dynamics, raw_data
        else:
            filename = os.path.basename(self.raw_file_paths[index+(self.cond_steps-1)*self.interval // self.data_freq])
            return latent_data, latent_dynamics, raw_data, filename
    
    def __del__(self):
        """Clean up thread pool"""
        if self._executor is not None:
            self._executor.shutdown()


class ERA5EDAIC(Dataset):
    def __init__(
        self,
        raw_root_dir,
        raw_variables,
        raw_transform,
        num_ensemble,
        steps,
        interval=6,
        data_freq=6,
        num_workers=16,
        return_filename=False,
    ):
        super().__init__()
        
        self.raw_root_dir = raw_root_dir
        self.raw_variables = raw_variables
        self.raw_transform = raw_transform
        self.num_ensemble = num_ensemble
        self.steps = steps
        self.interval = interval
        self.data_freq = data_freq
        self.return_filename = return_filename
        
        raw_file_paths = glob(os.path.join(raw_root_dir, '*.h5'))
        raw_file_paths = sorted(raw_file_paths)
        self.raw_inp_file_paths = raw_file_paths[:-(steps*interval // data_freq)]
        self.raw_file_paths = raw_file_paths
        
        # Create a single thread pool for the dataset
        self.num_workers = num_workers
        self._executor = None
        self._executor_lock = threading.Lock()
    
    @property
    def executor(self):
        """Lazy initialization of thread pool to ensure it's created in the correct thread"""
        if self._executor is None:
            with self._executor_lock:
                if self._executor is None:  # Double-check under lock
                    self._executor = ThreadPoolExecutor(max_workers=self.num_workers)
        return self._executor
    
    def _load_and_transform_frame(self, path):
        """Load and transform a single frame"""
        frame = get_data_given_path(path, self.raw_variables, stack_axis=1)[:self.num_ensemble] # N, C, H, W
        frame = torch.from_numpy(frame)
        frame = self.raw_transform(frame)
        return frame
    
    def __len__(self):
        return len(self.raw_inp_file_paths)
    
    def _getitem(self, file_paths, index, type):
        # Get paths for all frames in the sequence
        sequence_paths = [
            file_paths[index + (step * self.interval) // self.data_freq]
            for step in range(self.steps)
        ]
        
        # Submit all frame loading tasks to thread pool
        futures = [
            self.executor.submit(self._load_and_transform_frame, path)
            for path in sequence_paths
        ]
        
        # Collect results in order
        frames = [future.result() for future in futures]
        
        return torch.stack(frames, dim=1)  # NxTxCxHxW
    
    def __getitem__(self, index):
        raw_data = self._getitem(self.raw_file_paths, index, 'raw')
        if not self.return_filename:
            return raw_data
        else:
            filename = os.path.basename(self.raw_file_paths[index])
            return raw_data, filename
    
    def __del__(self):
        """Clean up thread pool"""
        if self._executor is not None:
            self._executor.shutdown()


class ERA5LatentForecastDataset(Dataset):
    def __init__(
        self,
        latent_root_dir,
        raw_root_dir,
        raw_variables,
        raw_transform,
        lead_times, # a list of lead times in hours
        data_freq=6,
        num_workers=16,
    ):
        super().__init__()
        
        self.latent_root_dir = latent_root_dir
        self.raw_root_dir = raw_root_dir
        self.raw_variables = raw_variables
        self.raw_transform = raw_transform
        self.lead_times = lead_times
        self.data_freq = data_freq
        
        raw_file_paths = glob(os.path.join(raw_root_dir, '*.h5'))
        raw_file_paths = sorted(raw_file_paths)
        max_lead_time = np.max(lead_times)
        self.raw_inp_file_paths = raw_file_paths[:-(max_lead_time // data_freq)]
        # raw_inp_file_paths = raw_file_paths[2:-(max_lead_time // data_freq)]
        # self.raw_inp_file_paths = []
        # for path in raw_inp_file_paths:
        #     filename = os.path.basename(path).split('.')[0]
        #     _, ic_idx = filename.split('_')
        #     ic_idx = int(ic_idx)
        #     if ic_idx % 2 == 0: # only use ICs initialized at 00 and 12 UTC
        #         self.raw_inp_file_paths.append(path)
        
        # build mapping from initial condition name to latent forecast paths and raw ground-truth paths
        self.ic_to_latent = {}
        self.ic_to_raw = {}
        for raw_ic_path in self.raw_inp_file_paths:
            ic_name = os.path.basename(raw_ic_path).split('.')[0]
            year, ic_idx = ic_name.split('_')
            ic_idx = int(ic_idx)
            self.ic_to_latent[ic_name] = []
            self.ic_to_raw[ic_name] = []
            for lead_time in lead_times:
                step = lead_time // data_freq
                
                latent_path = os.path.join(latent_root_dir, f'{ic_name}_to_{year}_{ic_idx + step:04d}.npy')
                assert os.path.exists(latent_path), f"Latent path {latent_path} does not exist"
                self.ic_to_latent[ic_name].append(latent_path)
                
                raw_path = os.path.join(raw_root_dir, f'{year}_{ic_idx + step:04d}.h5')
                assert os.path.exists(raw_path), f"Raw path {raw_path} does not exist"
                self.ic_to_raw[ic_name].append(raw_path)
        
        # Create a single thread pool for the dataset
        self.num_workers = num_workers
        self._executor = None
        self._executor_lock = threading.Lock()
    
    @property
    def executor(self):
        """Lazy initialization of thread pool to ensure it's created in the correct thread"""
        if self._executor is None:
            with self._executor_lock:
                if self._executor is None:  # Double-check under lock
                    self._executor = ThreadPoolExecutor(max_workers=self.num_workers)
        return self._executor
    
    def _load_latent(self, path):
        return torch.from_numpy(np.load(path))
    
    def _load_frame(self, path):
        """Load and transform a single frame"""
        frame = get_data_given_path(path, self.raw_variables)
        frame = torch.from_numpy(frame)
        frame = self.raw_transform(frame)
        return frame
    
    def __len__(self):
        return len(self.raw_inp_file_paths)
    
    def _getitem(self, file_paths, type):
        # Submit all frame loading tasks to thread pool
        load_func = self._load_latent if type == 'latent' else self._load_frame
        futures = [
            self.executor.submit(load_func, path)
            for path in file_paths
        ]
        
        # Collect results in order
        frames = [future.result() for future in futures]
        
        dim_stack = 1 if type == 'latent' else 0
        return torch.stack(frames, dim=dim_stack)  # TxCxHxW
    
    def __getitem__(self, index):
        ic_raw_path = self.raw_inp_file_paths[index]
        ic_name = os.path.basename(ic_raw_path).split('.')[0]
        latent_forecast_paths = self.ic_to_latent[ic_name]
        raw_ground_truth_paths = self.ic_to_raw[ic_name]
        latent_forecasts = self._getitem(latent_forecast_paths, 'latent')
        raw_ground_truths = self._getitem(raw_ground_truth_paths, 'raw')
        return latent_forecasts, raw_ground_truths
    
    def __del__(self):
        """Clean up thread pool"""
        if self._executor is not None:
            self._executor.shutdown()


# training dataset consists of 1 input and multiple desired outputs at multiple intervals (randomly chosen)
class ERA5MultiStepRandomizedDataset(Dataset):
    def __init__(
        self,
        root_dir,
        variables,
        inp_transform,
        out_transform_dict,
        steps,
        list_intervals=[6, 12, 24],
        data_freq=6,
    ):
        super().__init__()
        
        # intervals must be divisible by data_freq
        for l in list_intervals:
            assert l % data_freq == 0

        self.root_dir = root_dir
        self.variables = variables
        self.inp_transform = inp_transform
        self.out_transform_dict = out_transform_dict
        self.steps = steps
        self.list_intervals = list_intervals
        self.data_freq = data_freq
        
        file_paths = glob(os.path.join(root_dir, '*.h5'))
        file_paths = sorted(file_paths)
        self.inp_file_paths = file_paths[:-(steps * max(list_intervals) // data_freq)] # the last few points do not have ground-truth
        self.file_paths = file_paths
        
    def __len__(self):
        return len(self.inp_file_paths)
    
    def __getitem__(self, index):
        inp_path = self.inp_file_paths[index]
        inp_data = get_data_given_path(inp_path, self.variables)

        # randomly choose an interval and get the corresponding ground-truths
        chosen_interval = np.random.choice(self.list_intervals)
        year, inp_file_idx = os.path.basename(inp_path).split('.')[0].split('_')
        year, inp_file_idx = int(year), int(inp_file_idx)
        outs = []
        diffs = []
        last_out = inp_data
        
        # get ground-truths at multiple steps
        for step in range(1, self.steps + 1):
            out_path = self.file_paths[index + (step * chosen_interval) // self.data_freq]
            out = get_data_given_path(out_path, self.variables)
            diff = out - last_out
            diff = torch.from_numpy(diff)
            diffs.append(self.out_transform_dict[chosen_interval](diff))
            outs.append(out)
            last_out = out
        
        inp_data = torch.from_numpy(inp_data)
        diffs = torch.stack(diffs, dim=0)
        out_transform_mean = torch.from_numpy(self.out_transform_dict[chosen_interval].mean)
        out_transform_std = torch.from_numpy(self.out_transform_dict[chosen_interval].std)
        list_intervals = np.array([chosen_interval] * self.steps)
        list_intervals = torch.from_numpy(list_intervals).to(dtype=inp_data.dtype) / 10.0
        
        return (
            self.inp_transform(inp_data), # VxHxW
            diffs, # TxVxHxW
            out_transform_mean, # V
            out_transform_std, # V
            list_intervals,
            self.variables,
        )

# validation and test datasets consist of 1 input and multiple desired outputs at multiple lead times
class ERA5MultiLeadtimeDataset(Dataset):
    def __init__(
        self,
        root_dir,
        variables,
        transform,
        list_lead_times,
        data_freq=6,
    ):
        super().__init__()
        
        # lead times must be divisible by data_freq
        for l in list_lead_times:
            assert l % data_freq == 0

        self.root_dir = root_dir
        self.variables = variables
        self.transform = transform
        self.list_lead_times = list_lead_times
        self.data_freq = data_freq
        
        file_paths = glob(os.path.join(root_dir, '*.h5'))
        file_paths = sorted(file_paths)
        max_lead_time = max(*list_lead_times) if len(list_lead_times) > 1 else list_lead_times[0]
        max_steps = max_lead_time // data_freq
        self.inp_file_paths = file_paths[:-max_steps] # the last few points do not have ground-truth
        self.file_paths = file_paths
        
    def __len__(self):
        return len(self.inp_file_paths)
    
    def __getitem__(self, index):
        inp_path = self.inp_file_paths[index]
        inp_data = get_data_given_path(inp_path, self.variables)
        year, inp_file_idx = os.path.basename(inp_path).split('.')[0].split('_')
        year, inp_file_idx = int(year), int(inp_file_idx)
        dict_out = {}
        
        # get ground-truth paths at multiple lead times
        for lead_time in self.list_lead_times:
            out_path = self.file_paths[index + lead_time // self.data_freq]
            dict_out[lead_time] = get_data_given_path(out_path, self.variables)
            
        inp_data = torch.from_numpy(inp_data)
        dict_out = {lead_time: torch.from_numpy(out) for lead_time, out in dict_out.items()}
        
        return (
            self.transform(inp_data), # VxHxW
            {lead_time: self.transform(out) for lead_time, out in dict_out.items()},
            self.variables,
        )
