import numpy as np


def print_matrix(matrix, label="Matrix"):
    print(f"\n{label}:")
    rows, cols = matrix.shape
    col_widths = [max(len(f"{matrix[r][c]:.2f}") for r in range(rows)) for c in range(cols)]
    for r in range(rows):
        row_str = "  ".join(f"{matrix[r][c]:>{col_widths[c]}.2f}" for c in range(cols))
        print(f"  [ {row_str} ]")


def input_matrix(name):
    print(f"\n--- Enter {name} ---")
    while True:
        try:
            rows = int(input(f"Number of rows for {name}: "))
            cols = int(input(f"Number of columns for {name}: "))
            if rows <= 0 or cols <= 0:
                print("Rows and columns must be positive integers. Try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a whole number.")

    print(f"Enter the elements of {name} row by row, separated by spaces.")
    matrix_data = []
    for r in range(rows):
        while True:
            row_input = input(f"  Row {r + 1} ({cols} values): ").strip().split()
            if len(row_input) != cols:
                print(f"  Expected {cols} values, got {len(row_input)}. Try again.")
                continue
            try:
                row_values = [float(val) for val in row_input]
                matrix_data.append(row_values)
                break
            except ValueError:
                print("  Invalid number detected. Please enter numeric values only.")

    return np.array(matrix_data)


def matrix_addition(a, b):
    if a.shape != b.shape:
        print("\nError: Matrices must have the same dimensions for addition.")
        return None
    return a + b


def matrix_subtraction(a, b):
    if a.shape != b.shape:
        print("\nError: Matrices must have the same dimensions for subtraction.")
        return None
    return a - b


def matrix_multiplication(a, b):
    if a.shape[1] != b.shape[0]:
        print(f"\nError: Cannot multiply a {a.shape[0]}x{a.shape[1]} matrix with a {b.shape[0]}x{b.shape[1]} matrix.")
        print("Number of columns in Matrix A must equal number of rows in Matrix B.")
        return None
    return a @ b


def matrix_transpose(a):
    return a.T


def matrix_determinant(a):
    if a.shape[0] != a.shape[1]:
        print(f"\nError: Determinant requires a square matrix. Got shape {a.shape[0]}x{a.shape[1]}.")
        return None
    return np.linalg.det(a)


def matrix_inverse(a):
    if a.shape[0] != a.shape[1]:
        print(f"\nError: Inverse requires a square matrix. Got shape {a.shape[0]}x{a.shape[1]}.")
        return None
    det = np.linalg.det(a)
    if abs(det) < 1e-10:
        print("\nError: Matrix is singular (determinant is 0). Inverse does not exist.")
        return None
    return np.linalg.inv(a)


def show_menu():
    print("\n" + "=" * 50)
    print("        MATRIX OPERATIONS TOOL")
    print("=" * 50)
    print("  1. Addition           (A + B)")
    print("  2. Subtraction        (A - B)")
    print("  3. Multiplication     (A x B)")
    print("  4. Transpose          (A^T or B^T)")
    print("  5. Determinant        (|A| or |B|)")
    print("  6. Inverse            (A^-1 or B^-1)")
    print("  7. Re-enter matrices")
    print("  8. View current matrices")
    print("  9. Exit")
    print("=" * 50)


def select_matrix_for_unary_op(matrix_a, matrix_b):
    print("\nWhich matrix? (A / B): ", end="")
    choice = input().strip().upper()
    if choice == "A":
        return matrix_a, "A"
    elif choice == "B":
        return matrix_b, "B"
    else:
        print("Invalid choice.")
        return None, None


def main():
    print("Welcome to the Matrix Operations Tool!")
    print("This tool lets you perform Addition, Subtraction, Multiplication,")
    print("Transpose, Determinant, and Inverse operations on matrices using NumPy.")

    matrix_a = input_matrix("Matrix A")
    matrix_b = input_matrix("Matrix B")

    print_matrix(matrix_a, "Matrix A")
    print_matrix(matrix_b, "Matrix B")

    while True:
        show_menu()
        choice = input("Select an option (1-9): ").strip()

        if choice == "1":
            result = matrix_addition(matrix_a, matrix_b)
            if result is not None:
                print_matrix(result, "Result of A + B")

        elif choice == "2":
            result = matrix_subtraction(matrix_a, matrix_b)
            if result is not None:
                print_matrix(result, "Result of A - B")

        elif choice == "3":
            result = matrix_multiplication(matrix_a, matrix_b)
            if result is not None:
                print_matrix(result, "Result of A x B")

        elif choice == "4":
            selected, name = select_matrix_for_unary_op(matrix_a, matrix_b)
            if selected is not None:
                result = matrix_transpose(selected)
                print_matrix(result, f"Result of {name}^T (Transpose)")

        elif choice == "5":
            selected, name = select_matrix_for_unary_op(matrix_a, matrix_b)
            if selected is not None:
                result = matrix_determinant(selected)
                if result is not None:
                    print(f"\nDeterminant of {name} = {result:.4f}")

        elif choice == "6":
            selected, name = select_matrix_for_unary_op(matrix_a, matrix_b)
            if selected is not None:
                result = matrix_inverse(selected)
                if result is not None:
                    print_matrix(result, f"Result of {name}^-1 (Inverse)")

        elif choice == "7":
            matrix_a = input_matrix("Matrix A")
            matrix_b = input_matrix("Matrix B")
            print_matrix(matrix_a, "Matrix A")
            print_matrix(matrix_b, "Matrix B")

        elif choice == "8":
            print_matrix(matrix_a, "Matrix A")
            print_matrix(matrix_b, "Matrix B")

        elif choice == "9":
            print("\nThank you for using the Matrix Operations Tool. Goodbye!")
            break

        else:
            print("\nInvalid option. Please select a number between 1 and 9.")


if __name__ == "__main__":
    main()
