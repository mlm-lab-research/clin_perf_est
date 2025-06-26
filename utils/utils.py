import pickle as pkl
import os
import numpy as np
import pandas as pd
from scipy.stats import beta
from IPython.display import display
import logging

from utils.metrics import calculate_metrics, expected_calibration_error, root_brier_score
from utils.plots import *
from utils.performance_estimation_methods import *

def load_data_from_pkl(results_folder, name):
    """
    Loads the 'outs' and 'labs' pickle files for a given prefix 'name'
    from the specified 'results_folder'.

    Returns:
        tuple: (outs, labels) if available, else (None, None) for missing.
    """
    path_ = os.path.join(results_folder, f'{name}.pkl')

    data = None
    
    if os.path.exists(path_):
        with open(path_, 'rb') as f:
            data = pkl.load(f)
    else:
        print(f"{name}.pkl not found in {results_folder}")

    return data

def configure_logging(logger, output_path, name="logfile"):
    """Configures logging for a new dataset by creating a new log file."""
    # Remove old handlers to prevent duplicate logs
    while logger.hasHandlers():
        logger.handlers.clear()

    # Create new log file per dataset
    log_file = os.path.join(output_path, f"{name}.log")
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    # Console handler (optional, for debugging)
    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # Attach handlers
    logger.addHandler(file_handler)
    # logger.addHandler(console_handler)  # Comment this if you don't want console output

    # Set log level
    logger.setLevel(logging.INFO)

    logger.info(f"Logging configured for: {log_file}")
    print(f"Logging to {log_file}")  # Ensures visibility in notebooks

    return logger


############################################
#** Functions for MICCAI Paper Notebooks **#
############################################
def sample_beta_mixture(mixture_components, size, random_state=None):
    """
    Sample from a mixture of Beta distributions with reproducible randomness.

    Args:
        mixture_components (list of dict): Each dict has 'alpha', 'beta', and 'weight'.
        size (int): Total number of samples to draw.
        random_state (int or np.random.Generator, optional): Seed or RNG.

    Returns:
        np.ndarray: Samples drawn from the mixture, dtype float32.
    """
    # initialize RNG
    rng = np.random.default_rng(random_state)

    # determine how many samples per component
    weights = [c['weight'] for c in mixture_components]
    component_sizes = rng.multinomial(size, weights)

    # draw samples for each component
    samples = []
    for comp, n in zip(mixture_components, component_sizes):
        s = beta.rvs(comp['alpha'], comp['beta'], size=n, random_state=rng)
        samples.append(s)

    return np.concatenate(samples).astype(np.float32)


def resample_with_prevalence(y_true, p_desired, n_samples, random_state=42):
    rng = np.random.default_rng(random_state)
    
    y_true = np.asarray(y_true)
    pos_indices = np.where(y_true == 1)[0]
    neg_indices = np.where(y_true == 0)[0]
    
    n_pos = int(p_desired * n_samples)
    n_neg = int(n_samples - n_pos)
    
    if n_pos > len(pos_indices) or n_neg > len(neg_indices):
        raise ValueError("Not enough samples to achieve desired prevalence.")
    
    
    sampled_pos = rng.choice(pos_indices, size=n_pos, replace=False)
    sampled_neg = rng.choice(neg_indices, size=n_neg, replace=False)
    
    sampled_indices = np.concatenate([sampled_pos, sampled_neg])
    rng.shuffle(sampled_indices)
    
    return sampled_indices  # or return y_true[sampled_indices] if needed

class LabelShiftEvaluator:
    def __init__(self, keys, estimation_methods, shifts, total_samples, n_resamples, save_folder, optional_name='', logger=None):
        self.keys = keys # the metrics to be estimated
        self.estimation_methods = estimation_methods # methods used to estimate the metrics
        self.shifts = shifts # shift strengths
        self.total_samples = int(total_samples)
        self.n_resamples = n_resamples # How many resamples of the test set for each shift
        self.save_folder = save_folder # Folder to save the results
        self.optional_name = optional_name # Optional name for the results file
        self.columns = ['shift'] + [
            f'{method}_{metric}_{stat}' 
            for method in estimation_methods 
            for metric in keys 
            for stat in ['mean', 'std']
        ] + [f'{method}_{metric}_{stat}' for method in ['test', 'validation'] for metric in ['ece', 'rbs'] for stat in ['mean', 'std']]
        self.results_df = pd.DataFrame(index=range(len(shifts)), columns=self.columns)
        self.results_df['shift'] = shifts
        self.logger = logger


    def evaluate(self, test_labels, test_probs, val_labels, val_probs):
        for k, shift in enumerate(self.shifts):
            method_metrics = {m: {k: [] for k in self.keys} for m in self.estimation_methods} # For each methods, all metrics will be estimated
            calib_metrics = {'test': {'ece': [], 'rbs': []}, 'validation': {'ece': [], 'rbs': []}}

            for i in range(self.n_resamples):
                sampled_idx = resample_with_prevalence(test_labels, shift, self.total_samples, random_state=i)
                shifted_labels, shifted_probs = test_labels[sampled_idx], test_probs[sampled_idx]

                # Realized test & validation metrics
                test_m = calculate_metrics(shifted_labels, shifted_probs)
                test_m['ece'] = expected_calibration_error(shifted_labels, shifted_probs, adaptive=True)
                test_m['rbs'] = root_brier_score(shifted_labels, shifted_probs)

                val_m = calculate_metrics(val_labels, val_probs)
                val_m['ece'] = expected_calibration_error(val_labels, val_probs, adaptive=True)
                val_m['rbs'] = root_brier_score(val_labels, val_probs)

                for key in self.keys:  # append all except ece, rbs
                    method_metrics['test'][key].append(test_m[key])
                    method_metrics['validation'][key].append(val_m[key])
                for key in ['ece', 'rbs']:
                    calib_metrics['test'][key].append(test_m[key])
                    calib_metrics['validation'][key].append(val_m[key])

                estim_method_function_mapping = {'CBPE': calculate_CBPE_metrics,
                                                 'ATC': calculate_ATC_metrics,
                                                 'CMATC': CM_ATC_metric_estim,
                                                 'DoC': calculate_DoC_metrics,
                                                 'CMDoC': CM_DoC_metric_estim,
                                                 }

                for method in self.estimation_methods:
                    if method in estim_method_function_mapping:
                        # Call the corresponding function and append results
                        func = estim_method_function_mapping[method]
                        if method == 'CBPE':
                            method_metrics[method] = self._append_metrics(method_metrics[method], func(shifted_probs, ))
                        elif method in ['ATC', 'DoC']:
                            method_metrics[method] = self._append_metrics(method_metrics[method], func(val_probs.reshape(-1, 1), val_labels.reshape(-1, 1), shifted_probs.reshape(-1, 1)))
                        elif method in ['CMATC', 'CMDoC']:
                            method_metrics[method] = self._append_metrics(method_metrics[method], func(val_probs.reshape(-1, 1), val_labels.reshape(-1, 1), shifted_probs.reshape(-1, 1)))
                        else:
                            raise ValueError(f"Unknown estimation method: {method}")
                
                if i % max(1, self.n_resamples // 3) == 0:
                    print(f'Finished {i + 1} out of {self.n_resamples} resamples for shift {shift}')
                    if self.logger is not None:
                        self.logger.info(f'Finished {i + 1} out of {self.n_resamples} resamples for shift {shift}')

            # Store results
            self._store_results(k, shift, method_metrics, calib_metrics)

            if k % max(1, len(self.shifts) // 5) == 0:
                print(f'Finished {k + 1} out of {len(self.shifts)}')
                if self.logger is not None:
                    self.logger.info(f'Finished {k + 1} out of {len(self.shifts)} shifts')


        # Delete NaN columns after checking if there are any
        if self.results_df.isnull().values.any():
            print(f"Warning: NaN values found in results_df in col and row: {self.results_df.isnull().any(axis=0)}, {self.results_df.isnull().any(axis=1)}. Will be drpped")
            self.results_df = self.results_df.dropna(axis=1, how='all')
        
        print(f'saving results to {self.save_folder}')
        self.results_df.to_csv(os.path.join(self.save_folder, f'metrics_per_shift_{self.optional_name}.csv'), index=False)
        return self.results_df

    def _append_metrics(self, metric_dict, new_metrics):
        for key in new_metrics:
            metric_dict.setdefault(key, []).append(new_metrics[key])
        return metric_dict

    def _store_results(self, row_idx, shift, method_metrics, calib_metrics):
        self.results_df.loc[row_idx, 'shift'] = shift
        for method in self.estimation_methods:
            for key in self.keys:
                vals = method_metrics[method][key]
                self.results_df.loc[row_idx, f'{method}_{key}_mean'] = np.mean(vals)
                self.results_df.loc[row_idx, f'{method}_{key}_std'] = np.std(vals)
            if method in ['test', 'validation']:
                for m in ['ece', 'rbs']:
                    vals = calib_metrics[method][m]
                    self.results_df.loc[row_idx, f'{method}_{m}_mean'] = np.mean(vals)
                    self.results_df.loc[row_idx, f'{method}_{m}_std'] = np.std(vals)

class CovariateShiftEvaluatorCXR:
    def __init__(
        self,
        keys,
        estimation_methods,
        shift_ratios,
        total_samples,
        n_resamples,
        save_folder,
        bias_level,
        test_artifacts,
        TASK='Pleff',
        optional_name='',
        logger=None,
    ):
        """
        - keys: list of metric names (e.g. ['accuracy','precision',...])
        - estimation_methods: list of methods (e.g. ['test','CBPE','ATC',...])
        - shift_ratios: array of spurious ratios to sweep over
        - total_samples: number of samples per resample
        - n_resamples: how many times to resample at each ratio
        - save_folder: where to load your npy’s and dump results
        - bias_level: the BIAS_LEVEL used in your filenames
        - test_dataset: your Dataset object so we can extract the artifact column
        - TASK: the task name, used for loading the npy files
        """
        self.keys               = keys
        self.estimation_methods = estimation_methods
        self.shift_ratios       = shift_ratios
        self.total_samples      = int(total_samples)
        self.n_resamples        = n_resamples
        self.save_folder        = save_folder
        self.optional_name      = optional_name
        self.bias               = bias_level
        self.task               = TASK
        self.logger             = logger
        self.test_artifacts = test_artifacts

        # --- load test/val arrays as you did in the script ---
        synthetic_results_folder = '../results/'
        test_data = np.load(
            os.path.join(synthetic_results_folder, f'test_data_{TASK}_{bias_level}.npy'),
            allow_pickle=True,
        ).item()
        val_data = np.load(
            os.path.join(synthetic_results_folder, f'val_data_{TASK}_{bias_level}.npy'),
            allow_pickle=True,
        ).item()

        self.test_probs  = test_data['test_probs']
        self.test_labels = test_data['test_labels']
        self.val_probs   = val_data['val_probs']
        self.val_labels  = val_data['val_labels']

        self.test_artifacts = (self.test_artifacts.reshape(-1, 1))

        # overall class prevalence
        self.class_1_prevalence = float(self.test_labels.mean())

        # precompute the four subgroup indices & their probs
        mask = lambda cls, art: (
            (self.test_labels == cls) &
            (self.test_artifacts == art)
        )
        self.idx_1_1 = np.where(mask(1,1))[0]
        self.idx_1_0 = np.where(mask(1,0))[0]
        self.idx_0_1 = np.where(mask(0,1))[0]
        self.idx_0_0 = np.where(mask(0,0))[0]

        self.probs_1_1 = self.test_probs[self.idx_1_1]
        self.probs_1_0 = self.test_probs[self.idx_1_0]
        self.probs_0_1 = self.test_probs[self.idx_0_1]
        self.probs_0_0 = self.test_probs[self.idx_0_0]
        
        self.labels_1_1 = self.test_labels[self.idx_1_1]
        self.labels_1_0 = self.test_labels[self.idx_1_0]
        self.labels_0_1 = self.test_labels[self.idx_0_1]
        self.labels_0_0 = self.test_labels[self.idx_0_0]

        # Prepare the results DataFrame
        cols = ['shift'] + [
            f'{method}_{metric}_{stat}' 
            for method in estimation_methods 
            for metric in keys 
            for stat in ['mean', 'std']
        ] + [f'{method}_{metric}_{stat}' for method in ['test', 'validation'] for metric in ['ece', 'rbs'] for stat in ['mean', 'std']]
        self.results_df = pd.DataFrame(
            index=range(len(self.shift_ratios)),
            columns=cols,
        )
        self.results_df['shift'] = self.shift_ratios


    def evaluate(self):
        # mapping from method name → (func, argument‐builder)
        estim_map = {
            'test': {
                'func': calculate_metrics,
                'args': lambda sl, sp, ids: (sl, sp),
            },
            'validation': {
                'func': calculate_metrics,
                'args': lambda *_: (self.val_labels, self.val_probs),
            },
            'CBPE': {
                'func': calculate_CBPE_metrics,
                'args': lambda sl, sp, ids: (sp, False),
            },
            'ATC': {
                'func': calculate_ATC_metrics,
                'args': lambda sl, sp, ids: (
                    self.val_probs.reshape(-1,1),
                    self.val_labels.reshape(-1,1),
                    sp.reshape(-1,1),
                ),
            },
            'CMATC': {
                'func': CM_ATC_metric_estim,
                'args': lambda sl, sp, ids: (
                    self.val_probs.reshape(-1,1),
                    self.val_labels.reshape(-1,1),
                    sp.reshape(-1,1), 0.5,
                ),
            },
            'DoC': {
                'func': calculate_DoC_metrics,
                'args': lambda sl, sp, ids: (
                    self.val_probs.reshape(-1,1),
                    self.val_labels.reshape(-1,1),
                    sp.reshape(-1,1),
                ),
            },
            'CMDoC': {
                'func': CM_DoC_metric_estim,
                'args': lambda sl, sp, ids: (
                    self.val_probs.reshape(-1,1),
                    self.val_labels.reshape(-1,1),
                    sp.reshape(-1,1), 
                ),
            },

        }

        for i, ratio in enumerate(self.shift_ratios):
            # prepare container for this shift
            local_metrics = {
                m: {k: [] for k in self.keys}
                for m in self.estimation_methods
            }
            calib_metrics = {
                'test': {'ece': [], 'rbs': []},
                'validation': {'ece': [], 'rbs': []}
            }

            # compute group sizes once per ratio
            n1 = int(self.total_samples * self.class_1_prevalence)
            n0 = self.total_samples - n1

            # build the 4 subgroup sample counts
            n_1_1 = int(n1 * ratio)
            n_1_0 = n1 - n_1_1
            n_0_0 = int(n0 * ratio)
            n_0_1 = n0 - n_0_0
            for _ in range(self.n_resamples):
                #Sample indices
                idx_1_1 = np.random.choice(self.idx_1_1, size=n_1_1, replace=False)
                idx_1_0 = np.random.choice(self.idx_1_0, size=n_1_0, replace=False)
                idx_0_1 = np.random.choice(self.idx_0_1, size=n_0_1, replace=False)
                idx_0_0 = np.random.choice(self.idx_0_0, size=n_0_0, replace=False)

                # Resample
                shifted_labels = np.concatenate([
                    self.test_labels[idx_1_1],
                    self.test_labels[idx_1_0],
                    self.test_labels[idx_0_1],
                    self.test_labels[idx_0_0]
                ])
                shifted_probs = np.concatenate([
                    self.test_probs[idx_1_1],
                    self.test_probs[idx_1_0],
                    self.test_probs[idx_0_1],
                    self.test_probs[idx_0_0]
                ])
                ids = np.concatenate([idx_1_1, idx_1_0, idx_0_1, idx_0_0])
                
                # Realized test & validation metrics
                test_m = calculate_metrics(shifted_labels, shifted_probs)
                test_m['ece'] = expected_calibration_error(shifted_labels, shifted_probs, adaptive=True)
                test_m['rbs'] = root_brier_score(shifted_labels, shifted_probs)

                val_m = calculate_metrics(self.val_labels, self.val_probs)
                val_m['ece'] = expected_calibration_error(self.val_labels, self.val_probs, adaptive=True)
                val_m['rbs'] = root_brier_score(self.val_labels, self.val_probs)

                for key in self.keys:  # append all except ece, rbs
                    local_metrics['test'][key].append(test_m[key])
                    local_metrics['validation'][key].append(val_m[key])
                for key in ['ece', 'rbs']:
                    calib_metrics['test'][key].append(test_m[key])
                    calib_metrics['validation'][key].append(val_m[key])


                # compute each requested method
                for method in self.estimation_methods:
                    if method not in estim_map:
                        raise ValueError(f"Unknown estimation method: {method}")
                    func = estim_map[method]['func']
                    args = estim_map[method]['args'](shifted_labels, shifted_probs, ids)
                    res = func(*args)

                    for key in self.keys:
                        local_metrics[method][key].append(res[key])
                if _ % max(1, self.n_resamples//3) == 0:
                    print(f'Finished resample {_+1}/{self.n_resamples} for shift {ratio:.2f}')
                    if self.logger is not None:
                        self.logger.info(f'Finished resample {_+1}/{self.n_resamples} for shift {ratio:.2f}')

            # push one row of mean/std into results_df
            self._store_results(i, ratio, local_metrics, calib_metrics)

            if i % max(1, len(self.shift_ratios)//5) == 0:
                print(f'Finished {i+1}/{len(self.shift_ratios)}')
            if self.logger is not None:
                self.logger.info(f'Finished {i+1}/{len(self.shift_ratios)} shifts')

        # save and return
        out_path = os.path.join(
            self.save_folder,
            f'metrics_per_covshift_{self.task}_{self.optional_name}.csv',
        )
        self.results_df.to_csv(out_path, index=False)
        return self.results_df

    def _store_results(self, row_idx, shift, method_metrics, calib_metrics):
        self.results_df.loc[row_idx, 'shift'] = shift
        for method in self.estimation_methods:
            for key in self.keys:
                vals = method_metrics[method][key]
                self.results_df.loc[row_idx, f'{method}_{key}_mean'] = np.mean(vals)
                self.results_df.loc[row_idx, f'{method}_{key}_std'] = np.std(vals)
            if method in ['test', 'validation']:
                for m in ['ece', 'rbs']:
                    vals = calib_metrics[method][m]
                    self.results_df.loc[row_idx, f'{method}_{m}_mean'] = np.mean(vals)
                    self.results_df.loc[row_idx, f'{method}_{m}_std'] = np.std(vals)
