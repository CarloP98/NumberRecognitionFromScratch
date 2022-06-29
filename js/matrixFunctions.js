const operations = {
    sum: (a,b) => a+b,
    sub: (a,b) => a-b,
    mul: (a,b) => a*b,
    div: (a,b) => a/b,
    square: (a) => a**2,
    exp: (a) => Math.exp(a),
    log: (a) => Math.log(a),
    abs: (a) => Math.abs(a),
    tanh: (a) => Math.tanh(a),
    sqrt: (a) => Math.sqrt(a)
};

function matrixShape(m) {
    var dim = [];
    for (;;) {
        dim.push(m.length);
        if (Array.isArray(m[0])) m = m[0];
        else break;
    }
    return dim;
}

function onehot(x, size){
  var vec = new Array(size).fill(0);
  vec[x] = 1;
  return vec;
}

function transpose(m){
  return m[0].map((_, colIndex) => m.map(row => row[colIndex]));
}

function dotProd(m1, m2) {
    var result = [];
    for (var i = 0; i < m1.length; i++) {
        result[i] = [];
        for (var j = 0; j < m2[0].length; j++) {
            var sum = 0;
            for (var k = 0; k < m1[0].length; k++)
                sum += m1[i][k] * m2[k][j];
            result[i][j] = sum;
        }
    }
    return result;
}

function ewop(m1, m2, op) {
  var[s1,s2] = [matrixShape(m1), matrixShape(m2)]

  if(s1[0]==undefined && s2[0]==undefined)
    return operations[op](m1,m2);

  if(s2[0]==undefined){
    if(s1.length == 1){
      var result = [...m1];
      for(var row=0; row<result.length; row++)
        result[row] = operations[op](result[row], m2);
      return result
    }
    var matrix = m1.map(function(arr){return arr.slice();});
    for(var row=0; row<matrix.length; row++)
      for(var col=0; col<matrix[0].length; col++)
        matrix[row][col] = operations[op](matrix[row][col], m2);
    return matrix;
  }
  if(s1[0]==undefined){
    if(s2.length == 1){
      var result = [...m2];
      for(var row=0; row<result.length; row++)
        result[row] = operations[op](m1, result[row]);
      return result
    }
    var matrix = m2.map(function(arr){return arr.slice();});
    for(var row=0; row<matrix.length; row++)
      for(var col=0; col<matrix[0].length; col++)
        matrix[row][col] = operations[op](m1, matrix[row][col]);
    return matrix;
  }
  if(s1.length == 1 && s2.length==1){
    var result = [...m1];
    for(var row=0; row<result.length; row++)
      result[row] = operations[op](result[row], m2[row]);
    return result
  }
  if(s1[0] == 1 && s2[0]>1 && s1[1]==s2[1]){
    var result = m2.map(function(arr){return arr.slice();});
    for(var row=0; row<result.length; row++)
      for(var col=0; col<result[0].length; col++)
        result[row][col] = operations[op](result[row][col], m1[0][col]);
    return result;
  }
  if(s2[0] == 1 && s1[0]>1  && s1[1]==s2[1]){
    var result = m1.map(function(arr){return arr.slice();});
    for(var row=0; row<result.length; row++)
      for(var col=0; col<result[0].length; col++)
        result[row][col] = operations[op](result[row][col], m2[0][col]);
    return result;
  }
  if(s1[1] == 1 && s2[1]>1 && s1[0]==s2[0]){
    var result = m2.map(function(arr){return arr.slice();});
    for(var row=0; row<result.length; row++)
      for(var col=0; col<result[0].length; col++)
        result[row][col] = operations[op](result[row][col], m1[row][0]);
    return result;
  }
  if(s2[1] == 1 && s1[1]>1  && s1[0]==s2[0]){
    var result = m1.map(function(arr){return arr.slice();});
    for(var row=0; row<result.length; row++)
      for(var col=0; col<result[0].length; col++)
        result[row][col] = operations[op](result[row][col], m2[row][0]);
    return result;
  }
  else{
    var result = m1.map(function(arr){return arr.slice();});
    for(var row=0; row<result.length; row++)
      for(var col=0; col<result[0].length; col++)
        result[row][col] = operations[op](result[row][col], m2[row][col]);
    return result;
  }
}

function ewop1(m1, op) {
  if(m1.constructor !== Array)
    return operations[op](m1);
  s = matrixShape(m1)
  if(s.length == 1){
    var result = [...m1];
    for(var row=0; row<result.length; row++)
      result[row] = operations[op](result[row]);
    return result
  }
  var matrix = m1.map(function(arr){return arr.slice();});
  for(var row=0; row<matrix.length; row++)
    for(var col=0; col<matrix[0].length; col++)
      matrix[row][col] = operations[op](matrix[row][col]);
  return matrix;
}

function msum(m, axis=null){
  s = matrixShape(m);
  if(s.length==1)
    return m.reduce((ps, a) => ps + a, 0);
  if(axis==0)
     return [m.reduce((a, b) => a.map((x, i) => x + b[i]))]
  if(axis==1)
    return transpose([m.map(r => r.reduce((a, b) => a + b))])
  return m.reduce(function(a,b){return a.concat(b)}).reduce(function(a,b){return a+b});
}

function mmean(m, axis=null){
  var s = matrixShape(m)
  if(s.length==1)
    return m.reduce((ps, a) => ps + a, 0)/(s[0]);
  if(axis==0)
     return [m.reduce((a, b) => a.map((x, i) => x + b[i])/(s[0]*s[1]))]
  if(axis==1)
    return [m.map(r => r.reduce((a, b) => a + b)/(s[0]*s[1]))]
  return m.reduce(function(a,b){return a.concat(b)}).reduce(function(a,b){return a+b})/(s[0]*s[1]);
}

function max(m){
  s = matrixShape(m);
  if(s.length==1)
      return Math.max(...m);
  return Math.max(...m.map(r=>Math.max(...r)));
}

function reshape(m, shape) {
  r = shape[0];
  c = shape[1];
   if (r * c !== m.length * m[0].length)
      return m
   const res = []
   let row = []
   m.forEach(items => items.forEach((num) => {
      row.push(num)
      if (row.length === c) {
         res.push(row)
         row = []
      }
   }))
   return res
};

function eye(size, arr=[]){
  if(arr.length == 0 || arr.length != size)
    arr = new Array(size).fill(1);
   const res = [];
   for(let i = 0; i < size; i++){
      if(!res[i])
         res[i] = [];
      for(let j = 0; j < size; j++){
         if(i === j)
            res[i][j] = arr[i];
         else
            res[i][j] = 0;
      };
   };
   return res;
};

function argMax(arr) {
    if (arr.length === 0)
        return -1;
    var max = arr[0];
    var maxIndex = 0;
    for (var i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            maxIndex = i;
            max = arr[i];
        }
    }
    return maxIndex;
}

function randBetw(min,max){
    return Math.floor(Math.random()*(max-min+1)+min);
}

function assert(cond, text){
	if( cond )	return;
	if( console.assert.useDebugger )	debugger;
	throw new Error(text || "Assertion failed!");
}

function gaussianRandom() {
  var u = 0, v = 0;
  while(u === 0) u = Math.random();
  while(v === 0) v = Math.random();
  return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
}

function shuffleTwo(obj1, obj2) {
  var index = obj1.length;
  var rnd, tmp1, tmp2;

  while (index) {
    rnd = Math.floor(Math.random() * index);
    index -= 1;
    tmp1 = obj1[index];
    tmp2 = obj2[index];
    obj1[index] = obj1[rnd];
    obj2[index] = obj2[rnd];
    obj1[rnd] = tmp1;
    obj2[rnd] = tmp2;
  }
}


function sum(m1, m2){
  return ewop(m1, m2, "sum")
}
function sub(m1, m2){
  return ewop(m1, m2, "sub")
}
function mul(m1, m2){
  return ewop(m1, m2, "mul")
}
function div(m1, m2){
  return ewop(m1, m2, "div")
}
function exp(m1){
  return ewop1(m1, "exp")
}
function log(m1){
  return ewop1(m1, "log")
}
function square(m1){
  return ewop1(m1, "square")
}
function square(m1){
  return ewop1(m1, "abs")
}
function tanh(m1){
  return ewop1(m1, "tanh")
}
function sqrt(m1){
  return ewop1(m1, "sqrt")
}
