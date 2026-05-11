import subprocess
import hashlib
import json
import os
import sys
from datetime import datetime
import base64
import tkinter as tk
from tkinter import messagebox, ttk
from cryptography.fernet import Fernet

# MUST MATCH THE KEY IN generate_license.py
FERNET_KEY = b"iwBPZliqScYbmJuNRToQK8Y6LqtWs4FSgqCvuEG0_ik="

def get_hwid():
    """Generates a unique hardware ID for Windows."""
    try:
        cmd = "wmic csproduct get uuid"
        uuid = subprocess.check_output(cmd, shell=True).decode().split('\n')[1].strip()
        cmd = "wmic cpu get processorid"
        cpuid = subprocess.check_output(cmd, shell=True).decode().split('\n')[1].strip()
        raw_id = f"{uuid}-{cpuid}"
        return hashlib.sha256(raw_id.encode()).hexdigest().upper()[:16]
    except Exception:
        import platform
        raw_id = platform.node() + platform.processor()
        return hashlib.sha256(raw_id.encode()).hexdigest().upper()[:16]

def validate_license(license_path):
    """Validates the encrypted license file."""
    if not os.path.exists(license_path):
        return False, "LICENSE_MISSING", None

    try:
        with open(license_path, 'rb') as f:
            encrypted_data = f.read().strip()
        
        f_obj = Fernet(FERNET_KEY)
        decrypted_json = f_obj.decrypt(encrypted_data).decode()
        data = json.loads(decrypted_json)
        
        if data.get("hwid") != get_hwid():
            return False, "INVALID_HARDWARE", data
        
        expiry_date = datetime.strptime(data.get("expiry"), "%Y-%m-%d")
        if datetime.now() > expiry_date:
            return False, "LICENSE_EXPIRED", data
            
        return True, "VALID", data
    except Exception:
        return False, "TAMPERED_OR_INVALID", None

def show_activation_dialog(license_path):
    """Shows a Tkinter window to enter the license key."""
    root = tk.Tk()
    root.title("Face Search Activation")
    root.geometry("450x300")
    root.resizable(False, False)
    
    # Center the window
    root.eval('tk::PlaceWindow . center')

    hwid = get_hwid()
    
    style = ttk.Style()
    style.configure("TLabel", font=("Segoe UI", 10))
    style.configure("TButton", font=("Segoe UI", 10))

    frame = ttk.Frame(root, padding="20")
    frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(frame, text="Software Activation Required", font=("Segoe UI", 12, "bold")).pack(pady=(0, 10))
    
    ttk.Label(frame, text="Your Hardware ID (HWID):").pack(anchor=tk.W)
    hwid_entry = ttk.Entry(frame, font=("Consolas", 11), width=40)
    hwid_entry.insert(0, hwid)
    hwid_entry.config(state='readonly')
    hwid_entry.pack(pady=(0, 15))

    ttk.Label(frame, text="Enter License Key:").pack(anchor=tk.W)
    key_entry = ttk.Entry(frame, font=("Consolas", 10), width=40)
    key_entry.pack(pady=(0, 20))
    key_entry.focus()

    def on_activate():
        key = key_entry.get().strip()
        if not key:
            messagebox.showerror("Error", "Please enter a license key.")
            return
        
        try:
            # Try to save and validate
            with open(license_path, 'wb') as f:
                f.write(key.encode())
            
            ok, msg, _ = validate_license(license_path)
            if ok:
                messagebox.showinfo("Success", "Software activated successfully!")
                root.destroy()
                # We don't exit here, we let the main app continue
            else:
                os.remove(license_path) # Delete the bad key
                messagebox.showerror("Activation Failed", f"Invalid Key: {msg}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save license: {e}")

    btn_frame = ttk.Frame(frame)
    btn_frame.pack(fill=tk.X, pady=10)

    ttk.Button(btn_frame, text="Activate", command=on_activate).pack(side=tk.RIGHT, padx=5)
    ttk.Button(btn_frame, text="Exit", command=sys.exit).pack(side=tk.RIGHT)

    root.mainloop()

if __name__ == "__main__":
    # If run directly, just show the dialog for testing
    show_activation_dialog("license.key")
