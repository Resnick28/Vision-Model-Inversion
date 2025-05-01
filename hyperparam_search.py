# Privacy-Preserving Properties of Vision Classifiers

from generator_model import Generator
from utils import weights_initialization_gen_custom
from generator_trainer import ReconstructionTrainer
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective
from datetime import datetime
import matplotlib.pyplot as plt
import os
import sys
import contextlib

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def bayesian_optimization(mlp_model, device, dataset_info, train_loader, n_calls=100, 
                                  initial_points=None, validation_set=None):
    """
    Bayesian optimization with warmup and validation.
    """
    # Define search space
    # space = [
    #     Real(0.3, 1.5, name='alpha'),
    #     Real(0.5, 2.5, name='alpha_prime'),
    #     Real(3.0, 9.0, name='beta'),
    #     Real(4.0, 12.0, name='beta_prime'),
    #     Real(0.01, 0.3, name='gamma'),
    #     Real(0.01, 0.3, name='delta'),
    #     Real(0.01, 0.5, name='eta1'),
    #     Real(0.001, 0.05, name='eta2'),
    #     Real(0.001, 0.2, name='eta3')
    # ]
    space = [Real(0, 10, name=name) for name in ['alpha', 'alpha_prime', 'beta', 
                        'beta_prime', 'gamma', 'delta', 'eta1', 'eta2', 'eta3', 'eta4']]

    
    # Track all evaluations
    all_evaluations = []
    
    # Define the objective function that can use validation data if provided
    @use_named_args(space)
    def objective(alpha, alpha_prime, beta, beta_prime, gamma, 
                 delta, eta1, eta2, eta3, eta4):
        params = {
            'alpha': alpha,
            'alpha_prime': alpha_prime,
            'beta': beta,
            'beta_prime': beta_prime,
            'gamma': gamma,
            'delta': delta,
            'eta1': eta1,
            'eta2': eta2,
            'eta3': eta3,
            'eta4': eta4
        }
        generator_model = Generator().to(device)
        generator_model.apply(weights_initialization_gen_custom)
        trainer = ReconstructionTrainer(mlp_model, generator_model, device, dataset_info, train_loader=train_loader,
                    alpha=alpha, alpha_prime=alpha_prime, beta=beta, beta_prime=beta_prime, 
                    gamma=gamma, delta=delta, eta1=eta1, eta2=eta2, eta3=eta3, eta4=eta4)
        with suppress_output():
            trainer.train(num_epochs=50, batch_size=256, num_classes=10)

        best_ssim = trainer.ssim_list[-1][1]

        # Log the results for this iteration
        print(f"Hyperparams: α={alpha:.3f}, α'={alpha_prime:.3f}, β={beta:.3f}, β'={beta_prime:.3f}, "
              f"γ={gamma:.3f}, δ={delta:.3f}, η1={eta1:.3f}, η2={eta2:.4f}, η3={eta3:.4f}, η4={eta4:.4f}")
        print(f"SSIM: {best_ssim:.4f}")
        # print(f"Training accuracy: {train_acc:.4f}")
        # Store the evaluation
        all_evaluations.append((params, best_ssim))    
        
        return -best_ssim
    
    x0 = []
    y0 = []
    for params in initial_points:
        x_point = [params[dim.name] for dim in space]
        x0.append(x_point)

    # Run Bayesian optimization
    print("Starting Bayesian optimization...")
    
    # Run optimization with adaptive settings
    result = gp_minimize(
        objective, 
        space, 
        n_calls=n_calls,
        n_random_starts=max(5, n_calls//5),  # 20% random exploration
        x0=x0, 
        acq_func="EI",  # Expected Improvement acquisition function
        verbose=True
    )

     # Get the best hyperparameters
    best_hyperparams = {
        'alpha': result.x[0],
        'alpha_prime': result.x[1],
        'beta': result.x[2],
        'beta_prime': result.x[3],
        'gamma': result.x[4],
        'delta': result.x[5],
        'eta1': result.x[6],
        'eta2': result.x[7],
        'eta3': result.x[8],
        'eta4': result.x[9]
    }
    best_accuracy = -result.fun
    
    print("\n=== Best Hyperparameters Found ===")
    print(f"Best perturbation accuracy: {best_accuracy:.4f}")
    for param, value in best_hyperparams.items():
        print(f"{param}: {value:.6f}")

    # Create and save visualization of the optimization process
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_convergence(result, ax=ax)
    plt.savefig(f'bayesian_opt_convergence_{timestamp}.png')

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
    axes = axes.flatten()
    for i, (param_name, ax) in enumerate(zip([dim.name for dim in space], axes)):
        plot_objective(result, dimensions=[i], ax=ax)
        ax.set_title(f"Effect of {param_name}")
    plt.tight_layout()
    plt.savefig(f'bayesian_opt_importance_{timestamp}.png')

    return {
        'best_hyperparams': best_hyperparams,
        'best_accuracy': best_accuracy,
        'optimization_result': result
    }