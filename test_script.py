import subprocess
import pandas as pd
import re

import time
from datetime import datetime
import argparse

def execute_cpp_program(cpp_executable_path:str, args, output_csv:str, date_time:str, record_count:int = 1):
    try:
        start_time = time.time()
        command = [cpp_executable_path]  # Add the arguments to the command
        for args_key in args.keys():
            command += [f'--{args_key}',f'{args[args_key]}']

        args_df = pd.DataFrame({
            'datetime': [date_time]*args['frame'],
            'size': [args['size']]*args['frame'],
            'samples': [args['samples']]*args['frame'],
            'depth': [args['depth']]*args['frame'],
            'device': [args['device']]*args['frame'],
            'frame': [args['frame']]*args['frame'],
            'frame_index':list(range(args['frame']))
        })

        final_df = pd.DataFrame()

        for i in range(record_count):
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Regular expression to capture the frame time
            frame_time_regex = r"Frame time:\s*([0-9]+\.[0-9]+)\s*ms"

            output_lines = result.stdout.splitlines()
            frame_times = []

            for line in output_lines:
                match = re.search(frame_time_regex, line)
                if match:
                    frame_times.append(float(match.group(1)))

            df = pd.DataFrame(frame_times, columns=["frame_time"])
            combined_df = pd.concat([args_df, df], axis=1)
            final_df = pd.concat([final_df, combined_df],axis = 0)

            # Write the final DataFrame to a CSV file
        final_df.to_csv(output_csv, index=False, header=True)
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Program execution time: {elapsed_time:.6f} seconds")
        print(f"Extracted frame times written to {output_csv} successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    record_count = 1
    date_time= datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    csv_default_name = date_time.replace(':',"-",2)

    parser = argparse.ArgumentParser(description="Run a C++ program with arguments and capture output.")

    # Add the Python program's arguments
    parser.add_argument('--cpp_program_path', type=str, default=r'./RT_project.exe', dest='cpp_program_path', help="Cpp Program path (str)")
    parser.add_argument('--csv_path', type=str, default=f'{csv_default_name}.csv', dest='csv_path', help="Output file CSV path and name")

    parser.add_argument('--size', type=int, default=400, dest='size',help="Size parameter (int)")
    parser.add_argument('--samples', type=int, default=32, dest='samples', help="Samples parameter (int)")
    parser.add_argument('--depth', type=int, default=4, dest='depth', help="Depth parameter (int)")
    parser.add_argument('--device', type=str, default='gpu', dest='device', help="Device parameter (str)")
    parser.add_argument('--frame', type=int, default=62, dest='frame', help="Frame parameter (int)")

    args = parser.parse_args()

    cpp_program = args.cpp_program_path
    output_csv_file = args.csv_path

    args_list = {'size':args.size,'samples':args.samples,'depth':args.depth,'device':args.device, 'frame':args.frame}

    execute_cpp_program(cpp_program, args_list, output_csv_file, date_time, record_count)

if __name__ == '__main__':
    main()




