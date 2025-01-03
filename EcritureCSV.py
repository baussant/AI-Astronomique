import os
import pandas as pd 
import psutil
from psutil import process_iter


def fichier_ouvert(file_name):
    file_path = os.path.join(os.getcwd(),file_name)
    if os.path.isfile(file_path):
        return False
    else:
        return True

def find_and_kill_process(file_name):
    """
    Find which process is using the file and kill it.
    :param file_name: Path to the file
    """
    for proc in psutil.process_iter(['pid', 'name', 'open_files']):
        try:
            open_files = proc.info['open_files']
            if open_files:  # Vérifie si le processus est ouvert
                for f in open_files:
                    if file_name in f.path:  # Si le fichier est identique
                        print(f"Le processus {proc.info['name']} (PID: {proc.info['pid']}) utilise ce fichier.")
                        # Fermeture du processus windows
                        proc.terminate()
                        proc.wait()
                        print(f"Processus {proc.info['name']} a été fermé.")
                        return
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    print("Aucun processus n'utilise ce fichier")

def create_csv(data,file_name):
    # find_and_kill_process(file_name)
    df = pd.DataFrame(data)
    e = None
    try:
        df.to_csv(f"{file_name}", index=False,encoding='utf-8') 
            # df.to_excel(f"{file_name}.xlsx", index=False) # just for testing trop lent
    except Exception as e:
        find_and_kill_process(file_name)
        df.to_csv(f"{file_name}", index=False,encoding='utf-8')
        # print("Error : ",e)
        # file_path = os.path.join(os.getcwd(),file_name)

        return e     