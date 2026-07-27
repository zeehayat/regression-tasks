"""
Segment 16: Gradient Descent Loop Built By Hand & Divergence Breakage
Demonstrates stable convergence with lr = 0.01 and divergence with lr = 5.0
"""
import numpy as np

def run_gradient_descent(x_data, y_data, lr=0.01, steps=50):
    m, c = 0.0, 0.0
    history = []
    
    for step in range(1, steps + 1):
        # Compute predictions & error
        y_hat = m * x_data + c
        errors = y_data - y_hat
        loss = np.mean(errors**2)
        history.append((m, c, loss))
        
        # Compute gradients
        dm = -2.0 * np.mean(errors * x_data)
        dc = -2.0 * np.mean(errors)
        
        # Update parameters
        m = m - lr * dm
        c = c - lr * dc
        
        if step % 10 == 0 or step == 1:
            print(f"Step {step:2d} | m = {m:.4f}, c = {c:.4f} | Loss = {loss:.6f}")
            
    return m, c, history

if __name__ == "__main__":
    # MHP Dataset sample: Cable length (km) vs Cost (Million PKR)
    x_mhp = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float64)
    y_mhp = np.array([10.0, 18.0, 26.0, 34.0], dtype=np.float64) # True line: y = 4*x + 2
    
    print("--- Stable Gradient Descent Run (learning_rate = 0.01) ---")
    m_final, c_final, hist_stable = run_gradient_descent(x_mhp, y_mhp, lr=0.01, steps=50)
    initial_loss = hist_stable[0][2]
    final_loss = hist_stable[-1][2]
    print(f"Initial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")
    assert final_loss < initial_loss, "Expected loss to decrease in stable run"
    
    print("\n--- Mandatory Breakage: Exploding Gradient Descent (learning_rate = 5.0) ---")
    _, _, hist_exploding = run_gradient_descent(x_mhp, y_mhp, lr=5.0, steps=10)
    exploding_loss_start = hist_exploding[0][2]
    exploding_loss_end = hist_exploding[-1][2]
    print(f"Start Loss: {exploding_loss_start:.4f} -> End Loss: {exploding_loss_end:.4e}")
    assert exploding_loss_end > exploding_loss_start * 1e6, "Expected loop to diverge with lr=5.0"
    print("✅ ASSERTION PASSED: Stable lr=0.01 converges; lr=5.0 diverges as expected.")
