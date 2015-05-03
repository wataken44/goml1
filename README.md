
machine learning with golang
-----------------------------------------------------------

### run sample1(simple perceptron)
    cd $GOPATH/src
    git clone https://github.com/wataken44/goml1
    cd ../
    go install all
    ./bin/sample1

    create perceptron which predict 10x + 15y > 1 for (x,y)
    train with 10000 random data
    current w = [9.262579 12.667508 -1.000000 ]
    predict 10 random data
    x = -0.528983, y = 0.574850, exp = 1, act = 1, ok = true
    x = -0.065288, y = 0.598704, exp = 1, act = 1, ok = true
    x = -0.006957, y = -0.026327, exp = -1, act = -1, ok = true
    x = 0.758517, y = 0.423053, exp = 1, act = 1, ok = true
    x = 0.212782, y = 0.832552, exp = 1, act = 1, ok = true
    x = 0.206523, y = 0.718068, exp = 1, act = 1, ok = true
    x = -0.961382, y = -0.815997, exp = -1, act = -1, ok = true
    x = 0.908478, y = -0.419014, exp = 1, act = 1, ok = true
    x = -0.332744, y = -0.334723, exp = -1, act = -1, ok = true
    x = -0.015089, y = -0.309007, exp = -1, act = -1, ok = true
    
