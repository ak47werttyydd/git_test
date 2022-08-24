input_shapes = [[1, 3, 224, 224], [1, 6, 112, 112], [1, 12, 56, 56]]
output_shapes = [[1, 6, 112, 112], [1, 12, 56, 56], [1, 24, 28, 28]]
graph_gen_parameter = {
    "node_num":5, 
    "graph_mode":"ER", 
    "p":0.5
}
generate_cell_num = 24000
cell_store_location = "./cell"
cluster_center_num = 400
test_time = 7200