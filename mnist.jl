using MLDatasets: MNIST
using Flux

using LinearAlgebra
using Plots

using Images

using CUDA,cuDNN

using Random

using Statistics


nbrdata = 10_000 # the number of training data we will use from the MNIST dataset (if you have a bigger gpu (> 1050 Nvidia ) you can go bigger)
device = gpu # if you prefer cpu, change this to cpu (you should olso simplifie everything after this)

dataset = MNIST(:train) 
testset = MNIST(:test)

"""
This function preprocesses the data, meaning, we reshape the data from (28,28,L) to (28,28,1,L) to account for the convolutional layers,

also, we permute the first two dimensions of x_train, this is about how julia Images.jl works if one day you want to see the images,

finally, we transform the targets into one-hot encoding (L length vectors of classes (0 to 9) ) -> 10 by L Matrix of proba) 

note : there is a function in flux to one-hot encode arrays but *i thought it would be clearer here.

Send the data to gpu and return them.
"""
function prepocess(features,targets,L,device)
    x_train = features[:,:,1:L] .|> Float32
    x_train = reshape(x_train,28,28,1,L)
    x_train = permutedims(x_train,(2,1,3,4))

    yy = targets[1:L] .|> Float32
    y_train = zeros(Int,10,L)
    for i in 1:L
        j = floor(Int,yy[i])+1
        y_train[j,i] = 1
    end
    x_train = x_train |> device
    y_train = y_train |> device
    return x_train,y_train
end


features = dataset.features
targets = dataset.targets
x_train,y_train = prepocess(features,targets,nbrdata,device)
features_test = testset.features
targets_test = testset.targets
x_test,y_test = prepocess(features_test,targets_test,10_000,device)


"""
Here we define our neural network, mostly compose of convolutional and pooling layers.

architecture :

the x that will be input is of size (28,28,1,L), 

Conv((3,3),1=>4,tanh) will create 3*3= 9 filters of size 1*4=4 giving 9*4 = 36 parameters + 4 bias = 40 parameters with a tanh activation function,

the result (see convolution for further details) will be of size ((28-2),(28-2),4,L) = (26,26,4,L),

then MaxPool((2,2)) will reduce the size of the image to (26,26,4,L) = (13,13,4,L), mostly by summing over 2 by 2 submatrix,

then Conv((3,3),4=>8,tanh) will create 3*3= 9 filters of size 4*8=32 giving 9*32 = 288 parameters + 8 bias = 296 parameters with a tanh activation function,

the result will be of size ((13-2),(13-2),8,L) = (11,11,8,L),

again, MaxPool((2,2)) will reduce the size of the image to (11,11,8,L) = (5,5,8,L),

then we flatten the result into a vector of size (5*5*8,L) = (200,L),

then we add a dropout of 0.4 to avoid overfitting, (will set some coefs of the 200 vector to 0 with a probability of 0.4),

Next, we add a fully connected layer of size 50 (200*50 + 50 parameters = 10_050 parameters) with a tanh activation function,

the output will be of size (50,L) ,

and finally, another fully connected layer of size 10 (50*10 + 10 parameters = 510 parameters) with a softmax activation function

We get 10_896 parameters in total to optimise, it is actually quite a fin model compare to some used for MNIST, so don't expect too much from it.

"""

model = Chain(Conv((3,3),1=>4,tanh),
               MaxPool((2,2)),
               Conv((3,3),4=>8,tanh),
               MaxPool((2,2)),
               Flux.flatten,
               Flux.Dropout(0.4f0),
               Dense( (((28-2)÷2-2)÷2)^2 * 8 => 50,tanh),
               Dense(50=>10),
               Flux.softmax) |> device



model(x_train)


""" 
The loss used here is a crossentropy loss, won't explain it here, google it.
"""
loss(model,x,y)= Flux.crossentropy(model(x),y)

@show loss(model,x_test,y_test)

Flux.gradient(loss,model,x_test[:,:,:,1:1],y_test[:,1:1]);

"""
The optimiser used here is Adam with a learning rate of 0.01, again, won't explain it here. try Descent(0.01) if you want.
"""
opt = Flux.setup(Adam(0.01), model)

"""
This is where the magic goes, we train the model, it's really simple to read.
"""
function train_model(model,x,y,epochs,opt,bs)
    @assert 1 <= bs <= size(x,4)
    @inbounds for i in 1:epochs
        @show i,loss(model,x,y)
        data = Flux.DataLoader((x,y),batchsize = bs,shuffle=true) # collect data in batches (for 500 (batchsize in 10_000 (data size) will make an iterator of 20 elements of size 500 from a shuffle of the data)
        for d in data # get each 20 elements of size 500
            g = Flux.gradient(loss,model,d[1],d[2]) # d[1] : 500 values of x, d[2] : 500 values of y
            Flux.update!(opt, model, g[1]) # update the parameters from the gradient following an optimiser
        end
    end 
end


function evalClass(y)
    argmax.(eachcol(y)) .- 1
end


function confusion_matrix(Res)
    classes = unique(Res)
    cm = zeros(Int, length(classes), length(classes))
    for i in eachrow(Res)
        true_class = i[2] + 1  
        pred_class = i[1] + 1  
        cm[true_class, pred_class] += 1
    end
    return cm
end



function compute_metrics(cm)
    n = size(cm, 1)
    for i in 1:n
        tp = cm[i, i]
        fp = sum(cm[:, i]) - tp
        fn = sum(cm[i, :]) - tp
        tn = sum(cm) - tp - fp - fn

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)

        println("Class $(i-1): Precision = $precision, Recall = $recall, F1 Score = $f1")
    end
end

Flux.Optimisers.adjust!(opt,0.01)

# Little note here, if this feels really slow, first wait a second so your gpu wake up, otherwize, try lowering the number of points / change the model to get less parameters. it takes around 40s on my computer just to give you an idea
@time train_model(model,x_train,y_train,12,opt,500)

@show loss(model,x_test,y_test)


Res = hcat(evalClass(model(x_train) |> cpu ),evalClass(y_train |> cpu))
accuracy = mean(Res[:,1] .== Res[:,2])
cm = confusion_matrix(Res)
println("Confusion Matrix: \n", cm)
compute_metrics(cm)
Res = hcat(evalClass(model(x_test) |> cpu ),evalClass(y_test |> cpu))
accuracy = mean(Res[:,1] .== Res[:,2])
println("Accuracy: $accuracy")
cm = confusion_matrix(Res)
println("Confusion Matrix: \n", cm)
compute_metrics(cm)



