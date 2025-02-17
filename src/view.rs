use crate::value::Value;
use petgraph::dot::RankDir::LR;
use petgraph::dot::{Config, Dot};
use petgraph::graph::NodeIndex;
use petgraph::Graph;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug};
use std::fs::write;
use std::process::Command;
use std::rc::Rc;

#[derive(Debug, Clone)]
struct ValueGraph {
    nodes: Vec<Rc<Value>>,
    edges: Vec<(Rc<Value>, Rc<Value>)>,
}

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

pub fn print_computation_graph(root: Rc<Value>, output_path: Option<&str>) -> String {
    let mut graph = Graph::<NodeData, ()>::new();
    let value_graph = trace(root);
    let mut node_map = HashMap::with_capacity(value_graph.nodes.len());
    for node in &value_graph.nodes {
        let _node_id = graph.add_node(NodeData::new(
            format!(
                "{{ {} | data {} | grad {} }}",
                node.label, node.data, node.grad
            ),
            "record".to_string(),
        ));
        node_map.insert(node.uuid.to_string(), _node_id);
        if let Some(op) = node.op {
            let _op_id = graph.add_node(NodeData::new(format!("{}", op), "circle".to_string()));
            graph.add_edge(_op_id, _node_id, ());
            let mut op_key = node.uuid.to_string();
            op_key.push(op);
            node_map.insert(op_key, _op_id);
        }
    }
    for (n1, n2) in &value_graph.edges {
        let n1_key = n1.uuid.to_string();
        let mut n2_key = n2.uuid.to_string();
        n2_key.push(n2.op.unwrap());
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

/// Build a set of all nodes and edges in a graph.
fn trace(root: Rc<Value>) -> ValueGraph {
    let mut nodes = vec![];
    let mut edges = vec![];
    let mut visited = HashSet::new();
    fn build(
        v: Rc<Value>,
        nodes: &mut Vec<Rc<Value>>,
        edges: &mut Vec<(Rc<Value>, Rc<Value>)>,
        visited: &mut HashSet<uuid::Uuid>,
    ) {
        if !visited.contains(&v.uuid) {
            visited.insert(v.uuid);
            nodes.push(Rc::clone(&v));
            for child in &v.prev {
                let child = Rc::new(RefCell::take(child));
                edges.push((Rc::clone(&child), Rc::clone(&v)));
                build(Rc::clone(&child), nodes, edges, visited);
            }
        }
    }
    build(root, &mut nodes, &mut edges, &mut visited);
    ValueGraph { nodes, edges }
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
    use std::rc::Rc;

    #[test]
    fn test_print_computation_graph() {
        let a = Value::new(2.0, "a".to_string(), 0.0, vec![], None);
        let b = Value::new(-3.0, "b".to_string(), 0.0, vec![], None);
        let c = Value::new(10.0, "c".to_string(), 0.0, vec![], None);
        let mut e = a * b; // 6.0
        e.label = "e".to_string();
        let mut d = e + c;
        d.label = "d".to_string();
        let f = Value::new(-2.0, "f".to_string(), 0.0, vec![], None);
        let mut l = d * f;
        l.label = "L".to_string();

        assert_eq!(
            print_computation_graph(Rc::new(l), None),
            r#"digraph {
    rankdir="LR"
    0 [ label = "NodeData { label: \"{ L | data -8 | grad 1 }\", shape: \"record\" }" label="{ L | data -8 | grad 1 }" shape=record]
    1 [ label = "NodeData { label: \"*\", shape: \"circle\" }" label="*" shape=circle]
    2 [ label = "NodeData { label: \"{ d | data 4 | grad 1 }\", shape: \"record\" }" label="{ d | data 4 | grad 1 }" shape=record]
    3 [ label = "NodeData { label: \"+\", shape: \"circle\" }" label="+" shape=circle]
    4 [ label = "NodeData { label: \"{ e | data -6 | grad 1 }\", shape: \"record\" }" label="{ e | data -6 | grad 1 }" shape=record]
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
