from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import tkinter as tk
import os
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
np.seterr(all="ignore")
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib" )



Image.MAX_IMAGE_PIXELS = None  # Set to None to disable the limit
titles = ['CD138','HLAII','DAPI','CD10','Claudin1','MUC1']

terminate = False

def main():
    fp = '/project/mclark/For/Bayard/Ground_truthing/IGG4/cut_samples-TUBULES'

    def read_file_as_set(filename):
        with open(filename, 'r') as file:
            # Store the full paths exactly as they are in the file
            return {line.strip() for line in file}
    
    labeled_set = read_file_as_set('/project/mclark/For/Bayard/Ground_truthing/IGG4/labeled-TUBULES.txt')
    skipped_set = read_file_as_set('/project/mclark/For/Bayard/Ground_truthing/IGG4/hard_skip-TUBULES.txt')

    txtf = '/project/mclark/For/Bayard/Ground_truthing/IGG4/processed-TUBULES.txt'
    with open(txtf, 'r') as file:
        file_list = file.readlines()
    
    # Clean the file names (strip newlines and any extra spaces)
    file_list = [f.strip() for f in file_list]
    
    # Filter the files based on the labeled_set and skipped_set
    # Use full paths for comparison to ensure consistency
    valid_files = []
    for f in file_list:
        full_path = os.path.join(fp, f)
        if full_path not in labeled_set and full_path not in skipped_set:
            valid_files.append(f)

    random.shuffle(valid_files) # Note, picks random instance from list
    for random_file in valid_files:
        if terminate:
            return

        file_path = os.path.join(fp, random_file)
        cell_id = int(file_path.split('_')[4])  # Split and take the 3rd segment
        if cell_id == 0:
            continue

        data = np.load(file_path)
        root = tk.Tk()
        root.title("Image Survey")
        main_title = tk.Label(root, text="Ground Truth Labeling", font=("Arial", 16, "bold"))
        main_title.grid(row=0, column=0, columnspan=10, pady=10)
        def display_image(idx, image, row, col, title):
            # Normalize image to float between 0 and 1
            image_float = image.astype(np.float32) / np.max(image)  # Normalize to [0, 1] range

            rgb_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
            rgb_image[..., 1] = image_float  # Blue channel from the image
            if cell_id != 0:
                data[6][data[6] != int(cell_id)] = 0
            rgb_image[..., 0] =  data[6] 
            
            fig, ax = plt.subplots(figsize=(1.5,1.5)) 

            ax.imshow(rgb_image, cmap='gray', interpolation='nearest') # 0.78125
            ax.axis('off')
            canvas = FigureCanvasTkAgg(fig, master=root)
            canvas.draw()
            canvas.get_tk_widget().grid(row=row * 3 + 2, column=col * 2, columnspan=2, pady=5)
            plt.close(fig) 

        tuples_list = [
                (21,232), (171,232), (321,232),
                (21,447), (171,447), (321,447)
        ]
        tuple_to_int = {tup: idx for idx, tup in enumerate(tuples_list)}
        responses = [0] * 6

        def toggle_button(button_yes, button_no, action,index):
            x = button_yes.winfo_x()
            y = button_yes.winfo_y()

            coord_tuple = (x, y)
            # Retrieve the index based on the (x, y) tuple
            tuple_index = tuple_to_int.get(coord_tuple, "Tuple not found")
            (_,num) = index
            responses[tuple_index] = num
            if action == 'yes':
                button_yes.config(bg="green", relief=tk.SUNKEN)
                button_no.config(bg="lightgrey", relief=tk.RAISED)
            elif action == 'no':
                button_no.config(bg="red", relief=tk.SUNKEN)
                button_yes.config(bg="lightgrey", relief=tk.RAISED)

        for idx in range(6):
            image = data[idx]
            row, col = divmod(idx, 3)

            # Create title label for each image
            img_title = tk.Label(root, text=titles[idx], font=("Arial", 12, "bold"))
            img_title.grid(row=row * 3 + 1, column=col * 2, columnspan=2, pady=(7, 0))

            # Display the image
            display_image(idx, image, row, col, titles[idx])

            # Create Yes/No buttons for each image
            button_yes = tk.Button(root, text="Yes")
            button_no = tk.Button(root, text="No")

            # Attach toggle commands after buttons are defined
            button_yes.config(command=lambda b_yes=button_yes, b_no=button_no: toggle_button(b_yes, b_no, 'yes',(idx,1)))
            button_no.config(command=lambda b_yes=button_yes, b_no=button_no: toggle_button(b_yes, b_no, 'no',(idx,0)))

            # Place buttons in the grid layout
            button_yes.grid(row=row * 3 + 3, column=col * 2, padx=7, sticky="e")
            button_no.grid(row=row * 3 + 3, column=col * 2 + 1, padx=7, sticky="w")


        def on_submit():
            tag = file_path
            csv_path = "/project/mclark/For/Bayard/Ground_truthing/IGG4/manual_ground_truth-TUBULES.csv"
            df = pd.read_csv(csv_path)
            r = ['None'] + [tag] + responses + ['IGG4']

            df.loc[len(df)] = r
            df.to_csv(csv_path, index=False)
            with open("/project/mclark/For/Bayard/Ground_truthing/IGG4/labeled-TUBULES.txt", 'a') as file:
                file.write(f'{file_path}\n')
            root.destroy() 


        def on_next():
            with open("/project/mclark/For/Bayard/Ground_truthing/IGG4/hard_skip-TUBULES.txt", 'a') as file:
                file.write(f'{file_path}\n')
            root.destroy()            

        def on_exit():
            global terminate
            terminate = True
            root.destroy() 
            
        submit_button = tk.Button(root, text="Submit", command=on_submit)
        submit_button.grid(row=0, column=6, rowspan=2, padx=(15, 0), pady=10, sticky="ns")


        next_button = tk.Button(root, text="Next", command=on_next)
        next_button.grid(row=2, column=6, rowspan=2, padx=(15, 0), pady=10, sticky="ns")

        exit_button = tk.Button(root, text="Exit", command=on_exit)
        exit_button.grid(row=4, column=6, rowspan=2, padx=(15, 0), pady=10, sticky="ns")

        if not terminate:
            root.mainloop()


if __name__ == "__main__":
    main()
