import subprocess

subprocess.run(['python', '1_extract_R_channel.py'], check=True)
subprocess.run(['python', '2_rtv.py'], check=True)
subprocess.run(['python', '3_R_Channel_compensation.py'], check=True)
subprocess.run(['python', '4_FM_PCE.py'], check=True)