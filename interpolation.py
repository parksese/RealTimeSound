import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

class MultiParamTable:
    def __init__(self, csv_file):
        # Load CSV file
        df = pd.read_csv(csv_file)

        # Expected columns in the CSV:
        # spd, tilt, coll, aoa, a0, a1, b1, a2, b2
        # (You can add more columns if needed)
        self.param_cols = ["spd", "tilt", "coll", "aoa"]
        self.coeff_cols = ["a0", "a1", "b1", "a2", "b2"]

        # Convert to numpy arrays
        self.points = df[self.param_cols].values
        self.values = {col: df[col].values for col in self.coeff_cols}

        # Build interpolators
        self.interpolators = {}
        for coeff in self.coeff_cols:
            try:
                self.interpolators[coeff] = LinearNDInterpolator(self.points, self.values[coeff])
            except Exception as e:
                print(f"[Warning] Linear interpolation failed for {coeff}: {e}")
                print("Using nearest-neighbor interpolation instead.")
                self.interpolators[coeff] = NearestNDInterpolator(self.points, self.values[coeff])

        print(f"[INFO] Table loaded: {len(self.points)} rows, parameters = {self.param_cols}")

    def get_coeffs(self, spd, tilt, coll, aoa):
        """Interpolate coefficients at arbitrary condition"""
        query = np.array([[spd, tilt, coll, aoa]])
        result = {}
        for coeff in self.coeff_cols:
            val = self.interpolators[coeff](query)
            if np.isnan(val):
                # fallback if query outside convex hull
                val = self.values[coeff].mean()
            result[coeff] = float(val)
        return result


# ============== Example usage ==============
if __name__ == "__main__":
    # Example CSV file format:
    # spd,tilt,coll,aoa,a0,a1,b1,a2,b2
    # 54,20,35,0,-455.03,30.27,-144.29,-6.21,-0.45
    # 67,0,60,0,434.21,72.22,-903.17,-223.12,10.28
    # ...

    table = MultiParamTable("rotor_coeff_table.csv")

    # Query example:
    coeffs = table.get_coeffs(spd=50, tilt=30, coll=40, aoa=5)
    print("Interpolated coefficients:")
    for k, v in coeffs.items():
        print(f"  {k} = {v:.3f}")