## sobel filter using rust

to build and use it in python, call:

# way 1: using pure rust

```
cargo build --release
python python_with_rust.py img/lizard.jpg img/house.jpg img/ball.jpg
```

Results:

```txt
$ python python_with_rust.py img/lizard.jpg img/house.jpg img/ball.jpg
img/lizard.jpg with size 4000x4000
    Sobel done: 361.625ms
img/house.jpg with size 1920x1440
    Sobel done: 44.940ms
img/ball.jpg with size 200x169
    Sobel done: 966.400µs
```

# way 2: using open_cv native function:

```
python using_open_cv.py img/lizard.jpg img/house.jpg img/ball.jpg
```

Results:

```txt
$ python using_open_cv.py img/lizard.jpg img/house.jpg img/ball.jpg
total time: 0.04682469367980957
total time: 0.02669548988342285
total time: 0.002001523971557617
```

# way 3: using manual implementation of convolution in python:

```
python python_manual.py img/lizard.jpg img/house.jpg img/ball.jpg
```

Results:

```txt
$ python python_manual.py img/lizard.jpg img/house.jpg img/ball.jpg
total time: 1.2843925952911377
total time: 0.22237420082092285
total time: 0.004999876022338867
```
