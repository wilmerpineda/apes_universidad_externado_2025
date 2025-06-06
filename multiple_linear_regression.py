from typing import Dict, Optional, Union, Any
import numpy as np
import pandas as pd
from random import sample
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.outliers_influence import OLSInfluence
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class RegressionDiagnostics:
    """Class to store regression diagnostic results"""
    breusch_pagan_stat: float
    breusch_pagan_pvalue: float
    durbin_watson_stat: float
    durbin_watson_pvalue: float
    shapiro_stat: float
    shapiro_pvalue: float

@dataclass
class InfluentialPoints:
    """Class to store influential points analysis results"""
    leverage: np.ndarray
    high_leverage: np.ndarray
    cooks_distance: np.ndarray
    influential_by_cook: np.ndarray
    dffits: np.ndarray
    dfbetas: np.ndarray
    extreme_response: np.ndarray

class MultipleLinearRegression:
    """
    A class for performing multiple linear regression analysis with various estimation methods
    and diagnostic capabilities.
    """
    
    def __init__(self, method: str = 'ols', weights: Optional[np.ndarray] = None) -> None:
        """Initialize the regression model.

        Args:
            method (str, optional): Estimation method ('ols', 'mle', 'wls', 'rlm'). Defaults to 'ols'.
            weights (Optional[np.ndarray], optional): Weights for WLS estimation. Defaults to None.

        Raises:
            ValueError: If invalid method is provided.
        """
        valid_methods = {'ols', 'mle', 'wls', 'rlm'}
        if method not in valid_methods:
            raise ValueError(f"Invalid method. Choose from {valid_methods}")
        
        self.method = method
        self.weights = weights
        self.model = None
        self.results = None
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> None:
        """Fit the regression model to the data.
    
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Predictor variables matrix
            y (Union[np.ndarray, pd.Series]): Response variable vector
    
        Raises:
            ValueError: If weights are not provided for WLS method
        """
        # Store feature names if provided
        self.feature_names = None
        self.target_name = None
        
        # Handle pandas DataFrame/Series to preserve variable names
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_data = X.values
        else:
            X_data = X
            # Create default feature names if not provided
            self.feature_names = [f'X{i+1}' for i in range(X.shape[1])]
        
        if isinstance(y, pd.Series):
            self.target_name = y.name or 'y'
            y_data = y.values
        else:
            y_data = y
            self.target_name = 'y'
        
        # Add constant and create design matrix with named variables
        X_with_const = sm.add_constant(X_data)
        self.design_info = pd.DataFrame(
            X_with_const, 
            columns=['const'] + self.feature_names
        )
        
        model_mapping = {
            'ols': lambda: sm.OLS(y_data, self.design_info),
            'mle': lambda: sm.OLS(y_data, self.design_info),
            'wls': lambda: sm.WLS(y_data, self.design_info, weights=self.weights),
            'rlm': lambda: sm.RLM(y_data, self.design_info, M=sm.robust.norms.TukeyBiweight())
        }
        
        if self.method == 'wls' and self.weights is None:
            raise ValueError("Weights must be provided for WLS method.")
            
        self.model = model_mapping[self.method]()
        self.results = self.model.fit()
        
        # Update model results with variable names
        self.results.model.data.xnames = ['const'] + self.feature_names
        if self.target_name:
            self.results.model.data.ynames = self.target_name
 
    def summary(self) -> None:
        """
        Print comprehensive model summary statistics with variable names.
        
        Returns:
            None
        
        Raises:
            ValueError: If model has not been fitted yet.
        """
        if self.results is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Create a custom summary header with variable information
        print("\nMultiple Linear Regression Results")
        print("=" * 40)
        print(f"Method: {self.method.upper()}")
        print(f"Target Variable: {self.target_name}")
        print("\nFeature Names:")
        for i, name in enumerate(self.feature_names, 1):
            print(f"  {i}. {name}")
        print("\nModel Summary:")
        print("-" * 40)
        
        # Print the statsmodels summary
        print(self.results.summary())
 
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions using the fitted model.
    
        Args:
            X (Union[np.ndarray, pd.DataFrame]): Predictor variables matrix for prediction
    
        Returns:
            np.ndarray: Predicted values
    
        Raises:
            ValueError: If model has not been fitted
        """
        if self.results is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Convert input to appropriate format
        if isinstance(X, pd.DataFrame):
            X_data = X.values
        else:
            X_data = X
        
        X_with_const = sm.add_constant(X_data)
        return self.results.predict(X_with_const)
    
    def _plot_diagnostics(
        self, 
        residuals: np.ndarray, 
        max_points: int = 1000,
        plots: Optional[list] = None
    ) -> None:
        """Create diagnostic plots for regression analysis with optimized performance.
    
        Args:
            residuals (np.ndarray): Model residuals
            max_points (int, optional): Maximum number of points to plot. Defaults to 1000.
            plots (Optional[list], optional): List of plots to generate. 
                Options: ['residuals', 'qq', 'leverage', 'cooks']. 
                Defaults to all plots.
        """
        if plots is None:
            plots = ['residuals', 'qq', 'leverage', 'cooks']
    
        # Downsample for plotting if necessary
        n_samples = len(residuals)
        if n_samples > max_points:
            print(f"Downsampling from {n_samples} to {max_points} points for plotting...")
            idx = sample(residuals.index.to_list(),max_points)
            plot_residuals = residuals[idx]
            plot_fitted = self.results.fittedvalues[idx]
        else:
            plot_residuals = residuals
            plot_fitted = self.results.fittedvalues
    
        # Calculate influence measures only if needed
        if any(plot in ['leverage', 'cooks'] for plot in plots):
            influence = OLSInfluence(self.results)
            if n_samples > max_points:
                leverage = influence.hat_matrix_diag[idx]
                residuals_std = influence.resid_studentized[idx]
                cooks_d = influence.cooks_distance[0][idx]
            else:
                leverage = influence.hat_matrix_diag
                residuals_std = influence.resid_studentized
                cooks_d = influence.cooks_distance[0]
    
        # Determine number of subplots needed
        n_plots = len(plots)
        if n_plots == 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            axes = [ax]
        else:
            n_rows = (n_plots + 1) // 2
            fig, axes = plt.subplots(n_rows, min(2, n_plots), figsize=(12, 6*n_rows))
            if n_plots > 1:
                axes = axes.flatten()
    
        plot_idx = 0
    
        # Create only requested plots
        for plot_type in plots:
            if plot_type == 'residuals':
                print("Generating residuals plot...")
                axes[plot_idx].scatter(
                    plot_fitted, 
                    plot_residuals,
                    alpha=0.5,
                    s=20,
                    rasterized=True  # Improves performance for many points
                )
                axes[plot_idx].set_xlabel('Fitted Values')
                axes[plot_idx].set_ylabel('Residuals')
                axes[plot_idx].set_title('Residuals vs Fitted')
    
            elif plot_type == 'qq':
                print("Generating Q-Q plot...")
                # Use faster implementation for Q-Q plot
                sorted_residuals = np.sort(plot_residuals)
                theoretical_quantiles = stats.norm.ppf(
                    np.linspace(0.01, 0.99, len(sorted_residuals))
                )
                axes[plot_idx].scatter(
                    theoretical_quantiles,
                    sorted_residuals,
                    alpha=0.5,
                    s=20,
                    rasterized=True
                )
                axes[plot_idx].set_xlabel('Theoretical Quantiles')
                axes[plot_idx].set_ylabel('Sample Quantiles')
                axes[plot_idx].set_title('Normal Q-Q Plot')
    
            elif plot_type == 'leverage':
                print("Generating leverage plot...")
                axes[plot_idx].scatter(
                    leverage,
                    residuals_std,
                    alpha=0.5,
                    s=20,
                    rasterized=True
                )
                axes[plot_idx].axhline(y=0, color='gray', linestyle='--')
                axes[plot_idx].set_xlabel('Leverage')
                axes[plot_idx].set_ylabel('Studentized Residuals')
                axes[plot_idx].set_title('Leverage vs Studentized Residuals')
    
            elif plot_type == 'cooks':
                print("Generating Cook's distance plot...")
                # Only plot influential points
                threshold = 4 / len(residuals)
                significant_cooks = cooks_d > threshold
                if np.any(significant_cooks):
                    axes[plot_idx].stem(
                        np.where(significant_cooks)[0],
                        cooks_d[significant_cooks],
                        linefmt='r-',
                        markerfmt='ro'
                    )
                axes[plot_idx].set_xlabel('Observation Index')
                axes[plot_idx].set_ylabel("Cook's Distance")
                axes[plot_idx].set_title("Cook's Distance Plot (Significant Points)")
    
            plot_idx += 1
    
        plt.tight_layout()
        plt.show()
        plt.close()  # Explicitly close the figure to free memory
    
    def validate_assumptions(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        plot: bool = True,
        sample_size: Optional[int] = 5000,
        max_plot_points: int = 1000,
        plots: Optional[list] = None,
        random_state: int = 42
    ) -> RegressionDiagnostics:
        """Validate regression assumptions with optimized performance for large datasets.
    
        Args:
            X (np.ndarray): Predictor variables matrix
            y (np.ndarray): Response variable vector
            plot (bool, optional): Whether to create diagnostic plots. Defaults to True.
            sample_size (Optional[int], optional): Number of samples to use for tests. 
                If None, uses full dataset. Defaults to 5000.
            max_plot_points (int, optional): Maximum number of points to plot. Defaults to 1000.
            plots (Optional[list], optional): List of plots to generate. 
                Options: ['residuals', 'qq', 'leverage', 'cooks']. 
                Defaults to all plots.
            random_state (int, optional): Random seed for sampling. Defaults to 42.
    
        Returns:
            RegressionDiagnostics: Object containing test statistics and p-values
        """
         # Get residuals
        residuals = self.results.resid
        n_samples = len(residuals)
        
        # Sample data if needed
        if sample_size and n_samples > sample_size:
            print(f"Sampling {sample_size} records from {n_samples} for faster computation...")
            np.random.seed(random_state)
            sample_idx = [int(x) for x in list(np.random.choice(n_samples, sample_size, replace=False))]
            X_sample = X.loc[sample_idx]
            residuals_sample = residuals.loc[sample_idx]
            X_with_const = sm.add_constant(X_sample)
        else:
            X_with_const = sm.add_constant(X)
            residuals_sample = residuals
        
        print("Running statistical tests...")
        
        # Breusch-Pagan test (heteroscedasticity)
        print("- Computing Breusch-Pagan test...")
        bp_stat, bp_pvalue, _, _ = sm.stats.diagnostic.het_breuschpagan(
            residuals_sample, 
            X_with_const
        )
        
        # Durbin-Watson test (autocorrelation)
        print("- Computing Durbin-Watson test...")
        dw_stat = sm.stats.stattools.durbin_watson(residuals_sample)
        dw_pvalue = 2 * (1 - stats.norm.cdf(abs(dw_stat)))
        
        # Shapiro-Wilk test (normality)
        # For very large datasets, further sample for Shapiro-Wilk
        shapiro_size = min(len(residuals_sample), 5000)  # Shapiro-Wilk has a limit
        if len(residuals_sample) > shapiro_size:
            print(f"- Sampling {shapiro_size} records for Shapiro-Wilk test...")
            shapiro_sample = np.random.choice(residuals_sample, shapiro_size, replace=False)
        else:
            shapiro_sample = residuals_sample
        
        print("- Computing Shapiro-Wilk test...")
        shapiro_stat, shapiro_pvalue = shapiro_test = stats.shapiro(shapiro_sample)
    
        # Create diagnostic plots if requested
        if plot:
            print("\nGenerating diagnostic plots...")
            self._plot_diagnostics(
                residuals_sample, 
                max_points=max_plot_points,
                plots=plots
            )
        
        return RegressionDiagnostics(breusch_pagan_stat = float(bp_stat),
                                     breusch_pagan_pvalue = float(bp_pvalue),
                                     durbin_watson_stat = float(dw_stat),
                                     durbin_watson_pvalue = float(dw_pvalue),
                                     shapiro_stat = float(shapiro_stat),
                                     shapiro_pvalue = float(shapiro_pvalue)) 