## sobel filter using rust

implemented in 3 different ways, using pure rust, using native open cv functions and using python with manual convolution.

# way 1: using pure rust

```
cargo build --release
python python_with_rust.py img/lizard.jpg img/house.jpg img/ball.jpg
```

Results:

```txt
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
img/lizard.jpg with size 4000x4000
    Sobel done: 31.27312660217285ms
img/house.jpg with size 1440x1920
    Sobel done: 15.645742416381836ms
img/ball.jpg with size 169x200
    Sobel done: 0.9815692901611328ms
```

# way 3: using manual implementation of convolution in python:

```
python python_manual.py img/lizard.jpg img/house.jpg img/ball.jpg
```

Results:

```txt
img/lizard.jpg with size 4000x4000
    Sobel done: 1169.6667671203613ms
img/house.jpg with size 1440x1920
    Sobel done: 209.10310745239258ms
img/ball.jpg with size 169x200
    Sobel done: 4.987001419067383ms
```
