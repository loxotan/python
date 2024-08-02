import pandas as pd
import matplotlib.pyplot as plt

# Data for the compact table
data = {
    "Parameter": ["Δ PI (mean)", "Δ PI (deepest site)", "Δ GI (mean)", "Δ GI (deepest site)",
                  "Δ BOP (%) (mean)", "Δ BOP (%) (deepest)", "Δ PPD (mean, mm)", "Δ PPD (deepest, mm)",
                  "Δ RAL (mean, mm)", "Δ RAL (deepest, mm)", "Δ RGML (mean, mm)", "Δ RGML (deepest, mm)", "Δ TM"],
    "Group 1 (0-3 months)": ["1.30 ± 6.8", "1.58 ± 0.79", "1.36 ± 0.50", "1.47 ± 0.79",
                             "80.64 ± 16.81", "82.35 ± 39.29", "2.64 ± 0.76", "6.29 ± 1.10",
                             "2.15 ± 0.71", "5.29 ± 1.04", "-0.47 ± 0.49", "-1.00 ± 0.61", "12.41 ± 3.26"],
    "Group 2 (0-3 months)": ["1.47 ± 0.42", "1.61 ± 0.84", "1.31 ± 0.34", "1.27 ± 0.75",
                             "78.11 ± 17.07", "77.77 ± 42.77", "2.15 ± 0.70", "5.33 ± 1.13",
                             "1.52 ± 0.72", "4.11 ± 1.32", "-0.62 ± 0.42", "-1.22 ± 0.54", "10.55 ± 3.36"],
    "P Value (0-3 months)": ["0.491", "0.873", "0.708", "0.360", "0.678", "0.739", "0.07", "0.02*", "0.02*", "0.01*", "0.23", "0.27", "0.04*"],
    "Group 1 (0-6 months)": ["1.42 ± 0.64", "1.71 ± 0.59", "1.44 ± 0.47", "1.53 ± 0.72",
                             "83.52 ± 17.62", "82.35 ± 39.29", "3.31 ± 0.82", "7.52 ± 1.06",
                             "2.69 ± 0.79", "6.47 ± 1.07", "-0.60 ± 0.69", "-1.12 ± 0.78", "17.76 ± 4.51"],
    "Group 2 (0-6 months)": ["1.42 ± 0.51", "1.61 ± 0.69", "1.39 ± 0.44", "1.44 ± 0.70",
                             "79.83 ± 20.15", "83.33 ± 38.30", "3.41 ± 0.74", "7.22 ± 1.17",
                             "2.68 ± 0.84", "5.89 ± 1.37", "-0.75 ± 0.61", "-1.33 ± 0.69", "15.11 ± 4.33"],
    "P Value (0-6 months)": ["1", "0.77", "0.63", "0.65", "0.63", "0.94", "0.64", "0.54", "0.84", "0.22", "0.21", "0.33", "0.45"]
}

# Create DataFrame
df = pd.DataFrame(data)

# Plotting the table
fig, ax = plt.subplots(figsize=(15, 8))  # Set the size as needed
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', colColours=["lightgrey"]*len(df.columns))

# Save as an image file
output_path = "compact_table_image.jpg"
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.show()
