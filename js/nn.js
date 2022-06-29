const weightInitialize = (size) =>{
    return gaussianRandom()*Math.sqrt(2/size);
};

const activations = {
  sigmoid: (x) => div(1, sum(1,exp(mul(-1,x)))),
  relu: (x) => x.map(function(r){return r.map(function(c){return Math.max(1e-9,c);});}),
  tanh: (x) => tanh(x),
  softmax: (x) => {
    var e = exp(sub(x, max(x)))
    return div(e,msum(e,0))
  },
};

const activations_bp = {
  sigmoid: (dA, Z) => {
    var s =activations["sigmoid"](Z)
    return mul(dA,mul(s, sub(1,s)))
  },
  relu: (dA, Z) => dA.map((row,i)=>row.map((col,j)=>(Z[i][j]<=0)?0:col)),
  tanh: (x) => sub(1,square(tanh(x))),
  softmax: (dA, Z) => {
    return dA
    //var [m,n] = matrixShape(dA);
    //var p = activations["softmax"](Z);
    //var sm = sub(eye(n,p[0]),dotProd(transpose(p), p));
    //return dotProd(dA, sm);
  }
}

const costs ={
  "mse": (AL,Y)=> mmean(square(sub(Y,AL))),
  "bce": (AL,Y)=> -mmean(sum(mul(Y, log(AL)), mul(sub(1,Y), log(sub(1,AL))))),
  "cce": (AL,Y)=> -div(msum(mul(Y,log(sum(10e-10, AL)))),Y.length)
}

const costs_bp = {
  "mse": (AL,Y) => mul(-2/Y.length, sub(Y,AL)),
  "bce": (AL,Y) => mul(-1,sub(div(Y, AL),div(sub(1,Y),sub(1,AL)))),
  "cce": (AL,Y) => sub(AL,Y)
}

const regularizations = {
  "l1": (parameters, Y, lambda) =>{
    var regCost = 0
    for (const [key, value] of Object.entries(parameters))
      regCost += msum(abs(value))
    return lambda * regCost / (2 * matrixShape(Y)[1])
  },
  "l2": (parameters, Y, lambda) =>{
    var regCost = 0
    for (const [key, value] of Object.entries(parameters))
      regCost += msum(square(value))
    return lambda * regCost / (2 * matrixShape(Y)[1])
  },
}

const regularizations_bp = {
  "l1": (W, dW, lambda, m) => sum(div(lambda,m),div(lambda,m)),
  "l2": (W, dW, lambda, m) => div(sum(dW, mul(lambda, W)), m)
}

const optInitializers = {
  "adam": (layers) =>{
    var parameters = {}
    for(var i=1; i<layers;i++){
      parameters["Vdw"+i] = 0
      parameters["Vdb"+i] = 0
      parameters["Sdw"+i] = 0
      parameters["Sdb"+i] = 0
    }
    return parameters
  }
}

const optCompute = {
  "adam": (t, g, layer) => {
    var B1 = 0.9
    var B2 = 0.999
    var EPSILON = 10e-8
    t.optParameters["Vdw"+layer] = sum(mul(B1, t.optParameters["Vdw"+layer]),mul(1-B1, g["dW" + layer]), true)
    t.optParameters["Vdw"+layer] = sum(mul(B1, t.optParameters["Vdw"+layer]),mul(1-B1, g["dW" + layer]))
    t.optParameters["Vdb"+layer] = sum(mul(B1, t.optParameters["Vdb"+layer]),mul(1-B1, g["db" + layer]))
    t.optParameters["Sdw"+layer] = sum(mul(B2, t.optParameters["Sdw"+layer]),mul(1-B2, square(g["dW" + layer])))
    t.optParameters["Sdb"+layer] = sum(mul(B2, t.optParameters["Sdb"+layer]),mul(1-B2, square(g["db" + layer])))
    var VdwCorrected = div(t.optParameters["Vdw"+layer], 1-(B1**t.iterations))
    var VdbCorrected = div(t.optParameters["Vdb"+layer], 1-(B1**t.iterations))
    var SdwCorrected = div(t.optParameters["Sdw"+layer], 1-(B2**t.iterations))
    var SdbCorrected = div(t.optParameters["Sdb"+layer], 1-(B2**t.iterations))
    t.parameters["W" + layer] = sub(t.parameters["W" + layer], mul(t.learningRate, div(VdwCorrected, sum(sqrt(SdwCorrected),EPSILON))));
    t.parameters["b" + layer] = sub(t.parameters["b" + layer], mul(t.learningRate, div(VdbCorrected, sum(sqrt(SdbCorrected),EPSILON))));
  }
}

class nn {
    constructor(nnShape, costf, learningRate, other) {
    this.cache = [];
    this.iterations = 0;
    this.costf = costf;
    this.layers = nnShape;
    this.learningRate = learningRate;

    if(other != undefined){
      [this.reg, this.lambda] = (Array.isArray(other.regularizer) && other.regularizer.length ==2)?other.regularizer:[undefined, undefined];
      this.opt = other.optimizer
      this.optParameters = (this.opt == undefined)?undefined:optInitializers[this.opt](this.layers.length)
    }

    this.parameters = this.initializeParameters();
    };

    train(xs, ys, ret=""){
      this.iterations += 1
      var prediction = this.modelForward(xs)
      var cost = this.computeCost(prediction, ys)
      var grads = this.modelBackward(prediction, ys)
      this.updateParameters(grads)
      return cost
    }

    predict(xs){
        return this.modelForward(xs)
    }

    initializeParameters(nnShape){
        var parameters = {};
        for(var layer=1; layer < this.layers.length; layer++){
            parameters['W' + layer] = Array.from(Array(this.layers[layer][0]), () => Array.from(Array(this.layers[layer-1][0])).map(x=>weightInitialize(this.layers[layer-1][0])))
            parameters['b' + layer] = Array(this.layers[layer][0]).fill(Array(1).fill(0));
        }
        return parameters
    }

    modelForward(X){
        this.cache = []
        var prevLayer = transpose(X);
        for(var layer=1; layer < Math.floor(Object.keys(this.parameters).length/2)+1; layer++){
                 var W = this.parameters['W' + layer]
                 var b = this.parameters['b' + layer]
                 var Z = dotProd(W, prevLayer)
                 var Z = Z.map((row,i)=>row.map((col)=>col+parseFloat(b[i])))
                 this.cache.push([prevLayer, W, b, Z])
                 prevLayer = (this.layers[layer].length < 2)?Z:activations[this.layers[layer][1]](Z);
        }
        return prevLayer
    }

    modelBackward(AL, Y){
        var grads = {}
        var m = AL[0].length
        Y = transpose(Y)
        var dA_prev = costs_bp[this.costf](AL,Y)

        for(var layer=this.cache.length; layer>0; layer--){
            var [A_prev, W, b, Z] = this.cache[layer-1];
            var dZ = (this.layers[layer].length < 2)?dA_prev:activations_bp[this.layers[layer][1]](dA_prev, Z)
            var dW = mul(1/m, dotProd(dZ, transpose(A_prev)));

            if(this.reg !== undefined)
              dW = regularizations_bp[this.reg](W, dW, this.lambda, m)
            var db = mul(1/m, dZ.map(r => [r.reduce((a, b) => a + b)]));
            var dA_prev = dotProd(transpose(W), dZ);
            grads["dA" + (layer)] = dA_prev;
            grads["dW" + (layer)] = dW;
            grads["db" + (layer)] = db;
        }
        return grads
    }

    computeCost(AL, Y){
      if(this.reg !== undefined)
        return costs[this.costf](transpose(AL),Y) +  regularizations[this.reg](this.parameters, Y, this.lambda)
      return costs[this.costf](transpose(AL),Y);
    }

    updateParameters(grads){
        for(var layer=1; layer<Math.floor(Object.keys(this.parameters).length/2)+1; layer++){
            if(this.opt != undefined)
              optCompute[this.opt](this, grads, layer)
            else{
              this.parameters["W" + layer] = sub(this.parameters["W" + layer], mul(this.learningRate, grads["dW" + layer]));
              this.parameters["b" + layer] = sub(this.parameters["b" + layer], mul(this.learningRate, grads["db" + layer]));
            }
        }
    }
}
