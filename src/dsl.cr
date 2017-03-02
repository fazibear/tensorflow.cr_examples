require "tensorflow"

module TensorflowExamples

  graph = Tensorflow::Graph.new

  a = graph.constant(2.0)
  puts a

  #Tensorflow::Session.new(graph).run(a+b)

  #oper = Tensorflow::Operation.new(graph, "Const", "Dupa")
  #oper["Dupa"] = "dupa"
  #oper.finish

  #tensor = Tensorflow::Tensor.new("a")

end
