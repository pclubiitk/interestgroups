---
title: Image Processing using OpenCV
description: A tutorial on basic image processing and OpenCV library of C++
author: yashs15
date: 2016-05-28 17:48:35 +0530
categories: [Machine Learning, Computer Vision]
tags: [computer vision, tutorial]
pin: true
math: true
mermaid: true
image:
  path: /assets/img/OpenCV.png
  #url: https://3.bp.blogspot.com/-yvrV6MUueGg/ToICp0YIDPI/AAAAAAAAADg/SYKg4dWpyC43AAfrDwBTR0VYmYT0QshEgCPcBGAYYCw/s1600/OpenCV_Logo.png
  # lqip: data:image/webp;base64,UklGRpoAAABXRUJQVlA4WAoAAAAQAAAADwAABwAAQUxQSDIAAAARL0AmbZurmr57yyIiqE8oiG0bejIYEQTgqiDA9vqnsUSI6H+oAERp2HZ65qP/VIAWAFZQOCBCAAAA8AEAnQEqEAAIAAVAfCWkAALp8sF8rgRgAP7o9FDvMCkMde9PK7euH5M1m6VWoDXf2FkP3BqV0ZYbO6NA/VFIAAAA
  alt: OpenCV

---

OpenCV is the most widely used Computer Vision Library out there. It is
very powerful and offers many features suiting the needs of anyone requiring
Image Processing. This tutorial is a slightly modified version of
[this](https://students.iitk.ac.in/robocon/docs/doku.php?id=programming:image_processing_intro)
with a simpler method of compiling.

## Installation

The simplest possible install is always from the repositories.

For Debian/Ubuntu/Mint:

```
sudo apt-get install libopencv-dev
```

For Other distros like Arch, you can find the respective commands on the internet.

If you want python support too, you will need to install the python support by:

```
sudo apt-get install python-opencv
```

The repository versions of OpenCV are usually not the latest versions and you might miss
some features.

If you need specific features or a newer version of OpenCV (let's say for python3 support),
you could look [here](https://milq.github.io/install-opencv-ubuntu-debian/) for a very good guide
on compiling OpenCV from source.

## Compiling and running Programs in C++

Now that OpenCV has been installed, you might want to run a program to see if it is working
properly. I would suggest the following program:

``` cpp
#include <opencv2/opencv.hpp>
using namespace cv;
int main(int argc, char **argv) {

  // A camera object
  VideoCapture cam(0);

  // An image object
  Mat img;

  while(true) {

    // Storing a frame from camera to image
    cam >> img;

    // Do Your Operations Here

    // Displaying image in a window named "Image"
    imshow("Image",img);

    // Waiting for a keypress for 5ms

    /**
      * WaitKey is very important.
      * Even if you don't want a delay/keypress you should
      * use it with minimum delay otherwise, the gui does not update
      */
    int c = waitKey(5);

    // Quit on pressing q
    if( c == 'q' || c == 'Q')
      break;
  }
  return 0;
}
```

This simple program just takes input from your webcam and displays it in a window.
You can save it in a file named `test.cpp`.

Now, in order to compile this program, execute this from the terminal:

``` bash
g++ -o Test /path/to/test.cpp `pkg-config --libs opencv`
```

The parameter after `-o`, i.e., `Test` refers to the name of the executable created after compiling.

After compilation finishes (hopefully without any errors), you can run this test program with:

``` bash
./Test
```

For a more involved and complex compilation involving `cmake`, you can look
[here](https://students.iitk.ac.in/robocon/docs/doku.php?id=programming:image_processing_intro) and
[here](https://docs.opencv.org/2.4/doc/tutorials/introduction/linux_gcc_cmake/linux_gcc_cmake.html#linux-gcc-usage).

## Learning OpenCV

Now that you can compile and execute OpenCV programs, you are ready to begin learning OpenCV.
The OpenCV [tutorials](https://docs.opencv.org/2.4/doc/tutorials/tutorials.html) on their website is a
good starting point. My suggested order of reading is:

* [Load and display an image](https://docs.opencv.org/2.4/doc/tutorials/introduction/display_image/display_image.html#display-image)
* [Adding Images](https://docs.opencv.org/2.4/doc/tutorials/core/adding_images/adding_images.html#adding-images)
* [Basic Drawing](https://docs.opencv.org/2.4/doc/tutorials/core/basic_geometric_drawing/basic_geometric_drawing.html#drawing-1)
* Followed by all tutorials in the [Image Processing Module](https://docs.opencv.org/2.4/doc/tutorials/imgproc/table_of_content_imgproc/table_of_content_imgproc.html#table-of-content-imgproc)

The above order is because the initial tutorials tend to be more theoretical and focussing on the internals of OpenCV
rather than actual usage.

Plus, you might not need to see all the tutorials depending upon your needs.

Best of Luck developing your OpenCV application.