# RAFT

This repository contains the starter code for project 2 (15-440/15-640, Fall 2022). It also contains
some of the tests that we will use to grade your implementation.

## Starter Code

The starter code for this project is organized as follows:

```

src/github.com/cmu440/        
  raft/                            Raft implementation, tests and test helpers

  rpc/                             RPC library that must be used for implementing Raft

```

## Instructions

##### How to Write Go Code

If at any point you have any trouble with building, installing, or testing your code, the article
titled [How to Write Go Code](https://go.dev/doc/code) is a great resource for understanding
how Go workspaces are built and organized. You might also find the documentation for the
[`go` command](http://golang.org/cmd/go/) to be helpful. As always, feel free to post your questions
on Edstem.

### Executing the official tests

#### 1. Checkpoint

To run the checkpoint tests, run the following from the src/github.com/cmu440/raft/ folder

```sh
go test -run 2A
```

We will also check your code for race conditions using Go’s race detector:

```sh
go test -race -run 2A
```

#### 2. Full test

To execute all the tests, run the following from the src/github.com/cmu440/raft/ folder

```sh
go test
```

We will also check your code for race conditions using Go’s race detector:

```sh
go test -race
```

## Submission
Please disable or remove all debug prints regardless of whether you are using our logging
 framework or not before submitting to Gradescope. This helps avoid inadvertent failures, 
 messy autograder outputs and style point deductions. 


For both the checkpoint and the final submission, create `handin.zip` using the following 
command under the `p2/` directory, and then upload it to Gradescope. 
```
sh make_submit.sh
```

## Miscellaneous

### Reading the API Documentation

Before you begin the project, you should read and understand all of the starter code we provide.
To make this experience a little less traumatic (we know, it's a lot :P),
fire up a web server and read the documentation in a browser by executing the following commands:

1. Install `godoc` globally, by running the following command **outside** the `src/github.com/cmu440` directory:
```sh
go install golang.org/x/tools/cmd/godoc@latest
```
2. Start a godoc server by running the following command **inside** the `src/github.com/cmu440` directory:
```sh
godoc -http=:6060
```
3. While the server is running, navigate to [localhost:6060/pkg/github.com/cmu440](http://localhost:6060/pkg/github.com/cmu440) in a browser.
