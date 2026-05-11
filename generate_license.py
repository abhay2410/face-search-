import json
from datetime import datetime, timedelta
from cryptography.fernet import Fernet

# MUST MATCH THE KEY IN licensing.py
FERNET_KEY = b"iwBPZliqScYbmJuNRToQK8Y6LqtWs4FSgqCvuEG0_ik="

def generate_key(hwid, customer_name, days_valid=365):
    expiry_date = (datetime.now() + timedelta(days=days_valid)).strftime("%Y-%m-%d")
    
    data = {
        "hwid": hwid.strip().upper(),
        "customer": customer_name,
        "expiry": expiry_date,
        "issued_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    json_str = json.dumps(data)
    
    # Encrypt using Fernet
    f_obj = Fernet(FERNET_KEY)
    encrypted_key = f_obj.encrypt(json_str.encode())
    
    filename = f"license_{customer_name.replace(' ', '_')}.key"
    with open(filename, "wb") as f:
        f.write(encrypted_key)
    
    print(f"\n--- Encrypted License Generated ---")
    print(f"Customer: {customer_name}")
    print(f"HWID:     {hwid}")
    print(f"Expiry:   {expiry_date}")
    print(f"File:     {filename}")
    print(f"-----------------------------------\n")
    print("Instruction: Rename this file to 'license.key' and place it in the application folder.")

if __name__ == "__main__":
    print("=== License Generator (ADMIN ONLY) ===")
    h = input("Enter Client HWID: ")
    c = input("Enter Customer Name: ")
    d = input("Enter validity in days (default 365): ")
    days = int(d) if d.strip() else 365
    
    generate_key(h, c, days)
