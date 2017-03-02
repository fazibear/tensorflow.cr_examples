require "tensorflow"

module TensorflowExamples
  # new session
  deloc = ->(a : Pointer(Void), b : UInt64, c: Pointer(Void)) {}

  opts = LibTensorflow.new_session_options
  status = LibTensorflow.new_status
  session = LibTensorflow.new_deprecated_session(opts, status)

  puts LibTensorflow.get_code(status)

  # load graph

  file = File.read("./graph.pb")
  LibTensorflow.extend_graph(session, file, file.size, status)

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

  input_names = ["a", "b"]
  input_names_n = input_names.map(&.to_unsafe).to_unsafe

  output_names = ["c"]
  output_names_n = output_names.map(&.to_unsafe).to_unsafe

  target_names = [] of String
  target_names_n = target_names.map(&.to_unsafe).to_unsafe

  inputs = [a_tensor, b_tensor] of LibTensorflow::X_Tensor
  outputs = [c_tensor] of LibTensorflow::X_Tensor

  optss = LibTensorflow.new_buffer
  meta = LibTensorflow.new_buffer

  LibTensorflow.run(session, optss,
                    input_names_n,  inputs.to_unsafe,  input_names.size,
                    output_names_n, outputs.to_unsafe, output_names.size,
                    target_names_n, target_names.size,
                    meta, status)

  puts LibTensorflow.get_code(status)
  puts String.new(LibTensorflow.message(status))

  t = outputs[0]

  data = LibTensorflow.tensor_data(t)

  d = data.as(Float32*)
  puts d.value
end
