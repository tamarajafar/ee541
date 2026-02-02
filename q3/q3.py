import sys
import math

def secant_method(f, a, b):
    """
    Find the root of function f using the secant method.
    
    The secant method formula:
        x_n = x_{n-1} - f(x_{n-1}) * (x_{n-1} - x_{n-2}) / (f(x_{n-1}) - f(x_{n-2}))
    
    Convergence criterion: |x_{k+1} - x_k| < 10^-10
    
    Parameters:
        f: function to find root of
        a: left endpoint (becomes x_0)
        b: right endpoint (becomes x_1)

    """
    tolerance = 1e-10
    
    # Initialize: x_0 = a, x_1 = b
    x_prev = a      # x_{n-2} initially
    x_curr = b      # x_{n-1} initially
    
    history = [x_prev, x_curr]
    
    N = 0
    
    while True:
        # Evaluate function at current 2 points
        f_prev = f(x_prev)  # f(x_{n-2})
        f_curr = f(x_curr)  # f(x_{n-1})
        
        # Check for 0 denominator
        denominator = f_curr - f_prev
        if abs(denominator) < 1e-15:
            break
        
        # Apply secant method formula 
        # x_n = x_{n-1} - f(x_{n-1}) * (x_{n-1} - x_{n-2}) / (f(x_{n-1}) - f(x_{n-2}))
        x_next = x_curr - f_curr * (x_curr - x_prev) / denominator
        
        N += 1
        
        history.append(x_next)
        
        # Check convergence |x_{k+1} - x_k| < 10^-10
        if abs(x_next - x_curr) < tolerance:
            break
        
        x_prev = x_curr
        x_curr = x_next
        
        if N > 10000:
            break
    
    return N, history

def validate_inputs(a, b, f):
    """
    Validate inputs according to assignment requirements:
    1. a must be numeric (already checked by float conversion)
    2. b must be numeric (already checked by float conversion)
    3. a < b
    4. f(a) * f(b) < 0 (Bolzano's theorem - opposite signs)

    """
    # Check that a < b
    if a >= b:
        return False
    
    # Check Bolzano's theorem f(a) and f(b) must have opposite signs
    try:
        fa = f(a)
        fb = f(b)
        
        # If f(a) * f(b) >= 0, they have the same sign or one is zero
        # This means no root is guaranteed in [a, b]
        if fa * fb >= 0:
            return False
            
    except Exception:
        return False
    
    return True

def main():
    """
    Main function: parse command-line arguments and run secant method.
    """

    if len(sys.argv) != 3:
        print("Range error", file=sys.stderr)
        sys.exit(1)
    
    #  Convert arguments to floats 
    try:
        a = float(sys.argv[1])
        b = float(sys.argv[2])
    except ValueError:
        print("Range error", file=sys.stderr)
        sys.exit(1)
    
    #  Import the function f from func.py 
    try:
        from func import f
    except ImportError:
        print("Range error", file=sys.stderr)
        sys.exit(1)
    
    #  Validate
    if not validate_inputs(a, b, f):
        print("Range error", file=sys.stderr)
        sys.exit(1)
    
    # Run secant method 
    try:
        N, history = secant_method(f, a, b)
    except Exception:
        print("Range error", file=sys.stderr)
        sys.exit(1)

    print(N)
    print(f"{history[-3]:.15f}")
    print(f"{history[-2]:.15f}")
    print(f"{history[-1]:.15f}")

if __name__ == "__main__":
    main()
