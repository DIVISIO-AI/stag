#!/usr/bin/env python3

"""
STAG GUI
Simple GUI for Stephan's Automatic Image Tagger
"""

import os
import sys
import threading
import ctypes
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import webbrowser

import huggingface_hub
from huggingface_hub import hf_hub_download
from tktooltip import ToolTip

from stag import SKTagger, VERSION


class TextRedirector:
    """Redirects stdout/stderr to a tkinter Text widget."""
    
    def __init__(self, text_widget, tag="stdout"):
        """
        Initialize the redirector.
        
        Args:
            text_widget: The tkinter Text widget to redirect output to
            tag: The tag to apply to the text (default: "stdout")
        """
        self.text_widget = text_widget
        self.tag = tag

    def write(self, out_str):
        """Write text to the text widget."""
        self.text_widget.insert(tk.END, out_str, (self.tag,))
        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks()

    def flush(self):
        """Required for file-like objects."""
        pass


class StagGUI:
    """Main GUI class for the STAG application."""
    
    # Version is imported from stag module
    DEFAULT_PREFIX = "st"
    MODEL_REPO_ID = "xinyu1205/recognize-anything-plus-model"
    MODEL_FILENAME = "ram_plus_swin_large_14m.pth"
    
    def __init__(self, root):
        """
        Initialize the STAG GUI.
        
        Args:
            root: The tkinter root window
        """
        self.root = root
        self.stop_event = threading.Event()
        
        # Apply HiDPI scaling
        self.apply_hidpi_scaling()
        
        # Set up the UI
        self.root.title("DIVISIO STAG")
        self.setup_grid_configuration()
        self.load_images()
        self.create_widgets()
        
    def apply_hidpi_scaling(self):
        """Apply HiDPI scaling for better display on high-resolution screens."""
        try:
            # This function and the required win API calls are available on Windows 8.1+
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            # Get the DPI for the current window
            dpi = ctypes.windll.user32.GetDpiForWindow(self.root.winfo_id())
            # Calculate the scaling factor
            scaling_factor = dpi / 96
            # Set the scaling factor for the entire Tkinter application
            self.root.tk.call('tk', 'scaling', scaling_factor)
        except (AttributeError, TypeError):
            # Fallback for older Windows versions or other OS
            try:
                # This function is available on Windows Vista+
                ctypes.windll.user32.SetProcessDPIAware()
            except AttributeError:
                # If not on Windows, do nothing
                pass
    
    def setup_grid_configuration(self):
        """Configure the grid layout for responsive design."""
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(5, weight=1)
    
    def load_images(self):
        """Load and prepare images for the GUI."""
        img_dir = self.resource_path("images")
        
        # Load divisio logo
        divisio_logo_path = os.path.join(img_dir, "divisio_design-assets_logo_schwarz_WEB.png")
        self.divisio_logo_image = Image.open(divisio_logo_path)
        original_size = self.divisio_logo_image.size
        new_size = (int(original_size[0] * 0.5), int(original_size[1] * 0.5))
        self.divisio_logo_image = self.divisio_logo_image.resize(new_size)
        self.divisio_logo_photo = ImageTk.PhotoImage(self.divisio_logo_image)
        
        # Load stag logo
        stag_logo_path = os.path.join(img_dir, "stag_logo.png")
        self.stag_logo_image = Image.open(stag_logo_path)
        original_size = self.stag_logo_image.size
        new_size = (int(original_size[0] * 0.5), int(original_size[1] * 0.5))
        self.stag_logo_image = self.stag_logo_image.resize(new_size)
        self.stag_logo_photo = ImageTk.PhotoImage(self.stag_logo_image)
    
    def create_widgets(self):
        """Create and arrange all GUI widgets."""
        self.create_input_fields()
        self.create_checkboxes()
        self.create_buttons()
        self.create_output_area()
        self.create_branding()
    
    def create_input_fields(self):
        """Create the directory and prefix input fields."""
        # Image directory input
        ttk.Label(self.root, text="Image Directory:").grid(
            row=0, column=0, padx=5, pady=5, sticky=tk.W
        )
        self.entry_imagedir = ttk.Entry(self.root, width=50)
        self.entry_imagedir.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.browse_button = ttk.Button(
            self.root, text="Browse", command=self.browse_directory
        )
        self.browse_button.grid(row=0, column=2, padx=5, pady=5)
        
        # Prefix input
        ttk.Label(self.root, text="Prefix:").grid(
            row=1, column=0, padx=5, pady=5, sticky=tk.W
        )
        self.entry_prefix = ttk.Entry(self.root, width=50)
        self.entry_prefix.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.entry_prefix.insert(0, self.DEFAULT_PREFIX)
    
    def create_checkboxes(self):
        """Create checkbox options."""
        # Skip already tagged images checkbox
        self.var_skip = tk.BooleanVar(value=True)
        self.force_checkbox = ttk.Checkbutton(
            self.root, 
            text="Skip images already tagged by STAG", 
            variable=self.var_skip
        )
        self.force_checkbox.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        ToolTip(self.force_checkbox, msg=(
            "If this box is checked, STAG doesn't tag images which "
            "already have one or more tags with the given prefix."
        ))
        
        # Simulate tagging checkbox
        self.var_test = tk.BooleanVar()
        self.test_checkbox = ttk.Checkbutton(
            self.root, 
            text="Simulate tagging only", 
            variable=self.var_test
        )
        self.test_checkbox.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        ToolTip(self.test_checkbox, msg=(
            "Analyze images but don't write changes to the file system."
        ))
        
        # Darktable-compatible filenames checkbox
        self.var_prefer_exact_filenames = tk.BooleanVar()
        self.prefer_exact_filenames_checkbox = ttk.Checkbutton(
            self.root, 
            text="Use darktable-compatible filenames", 
            variable=self.var_prefer_exact_filenames
        )
        self.prefer_exact_filenames_checkbox.grid(
            row=2, column=2, padx=5, pady=5, sticky=tk.W
        )
        ToolTip(self.prefer_exact_filenames_checkbox, msg=(
            "When creating new XMP files, create PICT0001.JPG.XMP instead of PICT0001.XMP"
        ))
    
    def create_buttons(self):
        """Create action buttons."""
        # Run button
        self.run_button = ttk.Button(
            self.root, text="Run STAG", command=self.run_tagger
        )
        self.run_button.grid(row=3, column=0, pady=10)
        
        # Cancel button
        self.cancel_button = ttk.Button(
            self.root, text="Cancel", command=self.cancel_tagger
        )
        self.cancel_button.grid(row=3, column=1, pady=10)
        self.cancel_button.config(state='disabled')
    
    def create_output_area(self):
        """Create the tagger output text area."""
        ttk.Label(self.root, text="Tagger Output:").grid(
            row=4, column=0, columnspan=3, padx=5, pady=(10, 0), sticky=tk.W
        )
        
        # Text output with scrollbar
        text_frame = ttk.Frame(self.root)
        text_frame.grid(row=5, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")
        
        self.text_output = tk.Text(text_frame, height=15, wrap="word")
        self.text_output.grid(row=0, column=0, sticky="nsew")
        
        scrollbar = ttk.Scrollbar(text_frame, command=self.text_output.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.text_output['yscrollcommand'] = scrollbar.set
        
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
    
    def create_branding(self):
        """Create branding elements (logos and links)."""
        # STAG logo and version
        logo_frame = ttk.Frame(self.root)
        logo_frame.grid(row=6, rowspan=4, column=0, padx=5, pady=5, sticky='sw')
        
        stag_logo_label = ttk.Label(logo_frame, image=self.stag_logo_photo)
        stag_logo_label.pack()
        
        version_label = ttk.Label(logo_frame, text=f"Version {VERSION}")
        version_label.pack()
        
        # DIVISIO logo and info
        logo_label = ttk.Label(self.root, image=self.divisio_logo_photo)
        logo_label.grid(row=7, column=2, padx=5, pady=5, sticky='ne')
        
        creator_label = ttk.Label(self.root, text="Made with love by DIVISIO")
        creator_label.grid(row=8, column=2, padx=5, pady=5, sticky='ne')
        
        link = ttk.Label(
            self.root, 
            text="Visit our website", 
            foreground="blue", 
            cursor="hand2"
        )
        link.grid(row=9, column=2, padx=5, pady=5, sticky='ne')
        link.bind("<Button-1>", lambda e: self.open_webpage("https://divis.io"))
    
    def run_tagger(self):
        """Start the tagging process in a separate thread."""
        self.stop_event.clear()
        self.update_ui_state(running=True)

        imagedir = self.entry_imagedir.get()
        prefix = self.entry_prefix.get() or self.DEFAULT_PREFIX
        force = not self.var_skip.get()
        test = self.var_test.get()
        prefer_exact_filenames = self.var_prefer_exact_filenames.get()

        threading.Thread(
            target=self.run_tagger_thread, 
            args=(imagedir, prefix, force, test, prefer_exact_filenames)
        ).start()
    
    def run_tagger_thread(self, imagedir, prefix, force, test, prefer_exact_filenames):
        """Run the tagger in a separate thread."""
        # Redirect stdout and stderr to the text widget
        sys.stdout = TextRedirector(self.text_output, "stdout")
        sys.stderr = TextRedirector(self.text_output, "stderr")

        print("Starting tagger...")

        # Check if model was already downloaded
        dl_dir = os.path.join(
            huggingface_hub.constants.HF_HUB_CACHE,
            "models--xinyu1205--recognize-anything-plus-model"
        )
        if not os.path.isdir(dl_dir):
            self.show_startup_alert()
            print("First run – now downloading the model file.")
            print("This process can take a little while and is only executed once.")

        try:
            pretrained = hf_hub_download(
                repo_id=self.MODEL_REPO_ID, 
                filename=self.MODEL_FILENAME
            )

            tagger = SKTagger(
                pretrained, 
                384,  # Image size parameter
                force, 
                test, 
                prefer_exact_filenames, 
                prefix
            )

            if not self.stop_event.is_set():
                tagger.enter_dir(imagedir, self.stop_event)

            print("The mighty STAG has done its work. Have a nice day.")
        except Exception as e:
            print(f"Error during tagging: {e}")
        finally:
            # Reset UI state
            self.root.after(0, lambda: self.update_ui_state(running=False))
    
    def cancel_tagger(self):
        """Cancel the running tagger process."""
        print("Cancelling tagger...")
        self.stop_event.set()
    
    def browse_directory(self):
        """Open a file dialog to select an image directory."""
        directory = filedialog.askdirectory()
        if directory:
            self.entry_imagedir.delete(0, tk.END)
            self.entry_imagedir.insert(0, directory)
    
    def update_ui_state(self, running):
        """Update the UI state based on whether the tagger is running or not."""
        if running:
            for widget in (self.entry_imagedir, self.entry_prefix, self.browse_button,
                          self.run_button, self.force_checkbox, self.test_checkbox,
                          self.prefer_exact_filenames_checkbox):
                widget.config(state='disabled')
            self.cancel_button.config(state='normal')
        else:
            for widget in (self.entry_imagedir, self.entry_prefix, self.browse_button,
                          self.run_button, self.force_checkbox, self.test_checkbox,
                          self.prefer_exact_filenames_checkbox):
                widget.config(state='normal')
            self.cancel_button.config(state='disabled')
    
    def open_webpage(self, url):
        """Open a URL in the default web browser."""
        webbrowser.open_new(url)
    
    def resource_path(self, relative_path):
        """
        Get absolute path to resource, works for dev and for PyInstaller.
        
        Args:
            relative_path: Path relative to the script or PyInstaller bundle
            
        Returns:
            Absolute path to the resource
        """
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_path, relative_path)
    
    def show_startup_alert(self):
        """Show a message box about downloading the model on first run."""
        messagebox.showinfo(
            "Welcome to STAG", 
            "In order to be able to tag your images, STAG now needs to download "
            "the recognize-anything model from huggingface. This might take a while "
            "and is perfectly normal. The download is only done once, so the next "
            "time you start STAG you will be ready to go in an instant."
        )


def main():
    """Main entry point for the application."""
    root = tk.Tk()
    app = StagGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
