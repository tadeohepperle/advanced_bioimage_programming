## sobel filter using rust

to build and use it in python, call:

# way 1: using pure rust

```
cargo build --release
python index.py img/lizard.jpg img/house.jpg img/ball.jpg
```

# way 2: using open_cv native function:

```
python using_open_cv.py img/lizard.jpg img/house.jpg img/ball.jpg
```

# way 3: using manual implementation of convolution in python:

```
python python_manual.py img/lizard.jpg img/house.jpg img/ball.jpg
```
