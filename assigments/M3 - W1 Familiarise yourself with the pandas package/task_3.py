import pandas as pd
import matplotlib.pyplot as plt
import os 

def analyze_patient_survival(df):
    
    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Create the first scatter plot, with tumour size on the x-axis and patient survival on the y-axis
    df.plot.scatter(x='tumor_size', y='overall_survival_months', ax=ax1)
    # Set the title of the subplot
    ax1.set_title('tumor_size vs overall_survival_months')
    
    # Create the second scatter plot, with tumour stage on the x-axis and patient survival on the y-axis
    df.plot.scatter(x='tumor_stage', y='overall_survival_months', ax=ax2)
    # Set the title of the subplot
    ax2.set_title('tumor_stage vs overall_survival_months')
    
    # Create the third scatter plot, with Nottingham prognostic index on the x-axis and patient survival on the y-axis
    df.plot.scatter(x='nottingham_prognostic_index', y='overall_survival_months', ax=ax3)
    # Set the title of the subplot
    ax3.set_title('nottingham_prognostic_index vs overall_survival_months')
    
    # Adjust the spacing between the subplots
    plt.tight_layout()
    # Show the plot
    plt.show()
    
    """
    based on our plots:
    1)patients with greater tumor_size live less in comparison those with smaller tumor_size.
    2)patients with greater tumor_stage live less in comparison those with smaller tumor_stage.
    3)patients with greater nottingham_prognostic_index live less in comparison those with smaller nottingham_prognostic_index.
    """

# prject directory
current_folder = os.getcwd()
desktop_path = "C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad\\"
# desktop_fldr = os.path.join(user_fldr, "OneDrive - Probability\Bureaublad")

# define the filename
filename = "METABRIC_RNA_Mutation.csv"

# Read data using pandas
df = pd.read_csv(os.path.join(desktop_path, filename))
print(analyze_patient_survival(df))