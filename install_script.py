import subprocess
from rich import print

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while True:
        output = process.stdout.readline().decode()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    return rc

def main():
    print("[bold green]Installing necessary dependencies...[/bold green]")
    run_command("pip install numpy")
    run_command("apt-get install -y build-essential")
    run_command("apt-get install -y cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev")
    run_command("apt-get install -y libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev")
    
    print("[bold green]Downloading OpenCV source code...[/bold green]")
    run_command("git clone https://github.com/opencv/opencv.git")
    run_command("git clone https://github.com/opencv/opencv_contrib.git")

    print("[bold green]Configuring OpenCV build with CUDA support...[/bold green]")
    run_command("mkdir opencv/build")
    run_command("cd opencv/build && cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D INSTALL_PYTHON_EXAMPLES=ON \
        -D INSTALL_C_EXAMPLES=OFF \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D WITH_CUDA=ON \
        -D WITH_CUDNN=ON \
        -D OPENCV_DNN_CUDA=ON \
        -D ENABLE_FAST_MATH=1 \
        -D CUDA_FAST_MATH=1 \
        -D CUDA_ARCH_BIN=7.5 \
        -D WITH_CUBLAS=1 \
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        -D HAVE_opencv_python3=ON \
        -D BUILD_EXAMPLES=ON ..")

    print("[bold green]Compiling and installing OpenCV...[/bold green]")
    run_command("cd opencv/build && make -j8")
    run_command("cd opencv/build && make install")

    print("[bold green]Installation complete.[/bold green]")

if __name__ == '__main__':
    main()
