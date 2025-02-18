use crate::value::Value;
use petgraph::dot::RankDir::LR;
use petgraph::dot::{Config, Dot};
use petgraph::graph::NodeIndex;
use petgraph::Graph;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::write;
use std::process::Command;

#[derive(Debug, Clone)]
struct NodeData {
    label: String,
    shape: String,
}

impl NodeData {
    fn new(label: String, shape: String) -> Self {
        Self { label, shape }
    }
}

pub fn print_computation_graph(root: &Value, output_path: Option<&str>) -> String {
    let mut graph = Graph::<NodeData, ()>::new();
    let (nodes, edges) = root.trace();
    let mut node_map = HashMap::with_capacity(nodes.len());
    for node in &nodes {
        let _node_id = graph.add_node(NodeData::new(
            format!(
                "{{ {} | data {} | grad {} }}",
                node.label, node.data, node.grad
            ),
            "record".to_string(),
        ));
        node_map.insert(node.uuid.to_string(), _node_id);
        if let Some(op) = &node.op {
            let _op_id = graph.add_node(NodeData::new(op.to_string(), "circle".to_string()));
            graph.add_edge(_op_id, _node_id, ());
            let mut op_key = node.uuid.to_string();
            op_key += op;
            node_map.insert(op_key, _op_id);
        }
    }
    for (n1, n2) in &edges {
        let n1_key = n1.uuid.to_string();
        let mut n2_key = n2.uuid.to_string();
        let op = if let Some(op) = &n2.op { op } else { "" };
        n2_key += op;
        graph.add_edge(node_map[&n1_key], node_map[&n2_key], ());
    }

    let get_node_attrs = |_, node: (NodeIndex, &NodeData)| {
        format!("label=\"{}\" shape={}", node.1.label, node.1.shape)
    };

    let dot_string = format!(
        "{:?}",
        Dot::with_attr_getters(
            &graph,
            &[Config::EdgeNoLabel, Config::RankDir(LR)],
            &|_, _| {
                String::new() // No extra edge attributes
            },
            &get_node_attrs
        )
    );
    if let Some(path) = output_path {
        dot_to_svg(&dot_string, path);
    }

    dot_string
}

fn dot_to_svg(dot: &str, output_path: &str) {
    let dot_file = "graph.dot";
    write(dot_file, dot).expect("Failed to write DOT file");

    let output = Command::new("dot")
        .args(["-Tsvg", dot_file, "-o", output_path])
        .output()
        .expect("Failed to execute Graphviz");

    if output.status.success() {
        println!("SVG generated at {}", output_path);
    } else {
        eprintln!("Error: {}", String::from_utf8_lossy(&output.stderr));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_print_computation_graph() {
        let a = Value::new(2.0).with_label("a");
        let b = Value::new(-3.0).with_label("b");
        let c = Value::new(10.0).with_label("c");
        let e = (a * b).with_label("e"); // 6.0
        let d = (e + c).with_label("d");
        let f = Value::new(-2.0).with_label("f");
        let l = (d * f).with_label("L");

        assert_eq!(
            print_computation_graph(&l, None),
            r#"digraph {
    rankdir="LR"
    0 [ label = "NodeData { label: \"{ L | data -8 | grad 0 }\", shape: \"record\" }" label="{ L | data -8 | grad 0 }" shape=record]
    1 [ label = "NodeData { label: \"*\", shape: \"circle\" }" label="*" shape=circle]
    2 [ label = "NodeData { label: \"{ d | data 4 | grad 0 }\", shape: \"record\" }" label="{ d | data 4 | grad 0 }" shape=record]
    3 [ label = "NodeData { label: \"+\", shape: \"circle\" }" label="+" shape=circle]
    4 [ label = "NodeData { label: \"{ e | data -6 | grad 0 }\", shape: \"record\" }" label="{ e | data -6 | grad 0 }" shape=record]
    5 [ label = "NodeData { label: \"*\", shape: \"circle\" }" label="*" shape=circle]
    6 [ label = "NodeData { label: \"{ a | data 2 | grad 0 }\", shape: \"record\" }" label="{ a | data 2 | grad 0 }" shape=record]
    7 [ label = "NodeData { label: \"{ b | data -3 | grad 0 }\", shape: \"record\" }" label="{ b | data -3 | grad 0 }" shape=record]
    8 [ label = "NodeData { label: \"{ c | data 10 | grad 0 }\", shape: \"record\" }" label="{ c | data 10 | grad 0 }" shape=record]
    9 [ label = "NodeData { label: \"{ f | data -2 | grad 0 }\", shape: \"record\" }" label="{ f | data -2 | grad 0 }" shape=record]
    1 -> 0 [ ]
    3 -> 2 [ ]
    5 -> 4 [ ]
    2 -> 1 [ ]
    4 -> 3 [ ]
    6 -> 5 [ ]
    7 -> 5 [ ]
    8 -> 3 [ ]
    9 -> 1 [ ]
}
"#
        );
    }
}
