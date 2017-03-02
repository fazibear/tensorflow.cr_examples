require "tensorflow"

module TensorflowExamples
  # new session
  deloc = ->(a : Pointer(Void), b : UInt64, c: Pointer(Void)) {}

  opts = LibTensorflow.new_session_options
  status = LibTensorflow.new_status
  graph = LibTensorflow.new_graph
  session = LibTensorflow.new_session(graph, opts, status)

  puts "new_session:"
  puts LibTensorflow.get_code(status)

  # load graph

  file = File.read("./graph.pb")
  buffer = LibTensorflow.new_buffer_from_string(file, file.size)
  import_opts = LibTensorflow.new_import_graph_def_options
  LibTensorflow.graph_import_graph_def(graph, buffer, import_opts, status)

  puts "Load graph.."
  puts LibTensorflow.get_code(status)

  # init tensors

  a_dims = [] of Int64
  a_data = [3.0_f32] of Float32

  a_tensor = LibTensorflow.new_tensor(LibTensorflow::Datatype::Float,
                        a_dims, a_dims.size,
                        a_data, a_data.size,
                        deloc, nil)

  b_dims = [] of Int64
  b_data = [5.0_f32] of Float32

  b_tensor = LibTensorflow.new_tensor(LibTensorflow::Datatype::Float,
                        b_dims, b_dims.size,
                        b_data, b_data.size,
                        deloc, nil)

  c_dims = [] of Int64
  c_data = [] of Float32

  c_tensor = LibTensorflow.new_tensor(LibTensorflow::Datatype::Float,
                        c_dims, c_dims.size,
                        c_data, c_data.size,
                        deloc, nil)

  # run

  i1 = LibTensorflow::Output.new
  i1.oper = LibTensorflow.graph_operation_by_name(graph, "a")
  i1.index = 0

  i2 = LibTensorflow::Output.new
  i2.oper = LibTensorflow.graph_operation_by_name(graph, "b")
  i2.index = 0

  inputs = [i1, i2] of LibTensorflow::Output
  input_values = [a_tensor, b_tensor] of LibTensorflow::X_Tensor

  o1 = LibTensorflow::Output.new
  o1.oper = LibTensorflow.graph_operation_by_name(graph, "c")
  o1.index = 0

  outputs = [o1] of LibTensorflow::Output
  outputs_values = [c_tensor] of LibTensorflow::X_Tensor

  optss = LibTensorflow.new_buffer
  meta = LibTensorflow.new_buffer

  target = [] of LibTensorflow::X_Operation

  LibTensorflow.session_run(session, nil,
                    inputs, input_values, inputs.size,
                    outputs, outputs_values, outputs.size,
                    target, target.size,
                    nil, status)

  puts "Session run:"
  #puts LibTensorflow.get_code(status)
  #puts String.new(LibTensorflow.message(status))

  puts LibTensorflow.get_code(status)

  t = outputs_values[0]

  data = LibTensorflow.tensor_data(t)

  d = data.as(Float32*)
  puts d.value
end
