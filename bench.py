import subprocess
import os

if __name__ == "__main__":
    with open("benchmarks.txt", "w") as f:
        for filename in os.listdir("Generated_SpMV"):
            if filename.endswith(".c"):
                print(filename)
                subprocess.run(["gcc", "-O3", "-o", filename[:-2], filename], cwd="Generated_SpMV")
                subprocess.call(["./"+filename[:-2]], stdout=subprocess.PIPE, cwd="Generated_SpMV")
                sum = 0
                for i in range(5):
                    output = subprocess.run(["./"+filename[:-2]], capture_output=True, cwd="Generated_SpMV")
                    sum += float(output.stdout.decode("utf-8").split("\n")[1].split(" = ")[1])
                avg = sum/5
                f.write(f"{filename[:-2]}: {avg}ms\n")