use petgraph::dot::RankDir::LR;
use petgraph::dot::{Config, Dot};
use petgraph::graph::NodeIndex;
use petgraph::Graph;
use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::{Debug, Display, Formatter, Result};
use std::fs::write;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Mul};
use std::process::Command;
use std::rc::Rc;

#[derive(Clone)]
pub struct Value {
    pub label: String,
    uuid: uuid::Uuid,
    data: f64,
    grad: f64,
    prev: HashSet<Rc<Value>>,
    op: Option<char>,
}

#[derive(Debug, Clone)]
struct ValueGraph {
    nodes: BTreeSet<Rc<Value>>,
    edges: BTreeSet<(Rc<Value>, Rc<Value>)>,
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
    let mut nodes = BTreeSet::new();
    let mut edges = BTreeSet::new();
    fn build(
        v: Rc<Value>,
        nodes: &mut BTreeSet<Rc<Value>>,
        edges: &mut BTreeSet<(Rc<Value>, Rc<Value>)>,
    ) {
        if !nodes.contains(&v) {
            nodes.insert(Rc::clone(&v));
            for child in &v.prev {
                edges.insert((Rc::clone(child), Rc::clone(&v)));
                build(Rc::clone(child), nodes, edges);
            }
        }
    }
    build(root, &mut nodes, &mut edges);
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

impl Value {
    pub fn new(
        data: f64,
        label: String,
        grad: f64,
        prev: HashSet<Rc<Value>>,
        op: Option<char>,
    ) -> Self {
        Self {
            label,
            uuid: uuid::Uuid::new_v4(),
            data,
            grad,
            prev,
            op,
        }
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}:{}", self.label, self.data)
    }
}
impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "data={}", self.data)
    }
}

impl Add for Value {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.data + rhs.data,
            "".to_string(),
            0.0,
            vec![Rc::new(self), Rc::new(rhs)].into_iter().collect(),
            Some('+'),
        )
    }
}

impl Mul for Value {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.data * rhs.data,
            "".to_string(),
            0.0,
            vec![Rc::new(self), Rc::new(rhs)].into_iter().collect(),
            Some('*'),
        )
    }
}

impl PartialEq for Value {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.uuid == other.uuid
    }
}

impl Eq for Value {}

impl PartialOrd for Value {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }

    #[inline]
    fn lt(&self, other: &Self) -> bool {
        !self.data.ge(&other.data)
    }

    #[inline]
    fn le(&self, other: &Self) -> bool {
        other.data.ge(&self.data)
    }

    #[inline]
    fn gt(&self, other: &Self) -> bool {
        !other.data.ge(&self.data)
    }

    #[inline]
    fn ge(&self, other: &Self) -> bool {
        // We consider all NaNs equal, and NaN is the largest possible
        // value. Thus if self is NaN we always return true. Otherwise
        // self >= other is correct. If other is also not NaN it is trivially
        // correct, and if it is we note that nothing can be greater or
        // equal to NaN except NaN itself, which we already handled earlier.
        self.data.is_nan() | (self.data >= other.data)
    }
}

impl Ord for Value {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.data
            .partial_cmp(&other.data)
            .unwrap_or(Ordering::Equal) // Handle NaN safely
            .then_with(|| self.uuid.cmp(&other.uuid))
    }
}

impl Hash for Value {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.uuid.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;

    #[test]
    fn test_print_computation_graph() {
        let a = Value::new(2.0, "a".to_string(), 0.0, HashSet::new(), None);
        let b = Value::new(-3.0, "b".to_string(), 0.0, HashSet::new(), None);
        let c = Value::new(10.0, "c".to_string(), 0.0, HashSet::new(), None);
        let mut e = a * b; // 6.0
        e.label = "e".to_string();
        let mut d = e + c;
        d.label = "d".to_string();
        let f = Value::new(-2.0, "f".to_string(), 0.0, HashSet::new(), None);
        let mut l = d * f;
        l.label = "L".to_string();

        assert_eq!(
            print_computation_graph(Rc::new(l), None),
            r#"digraph {
    rankdir="LR"
    0 [ label = "NodeData { label: \"{ L | data -8 | grad 0 }\", shape: \"record\" }" label="{ L | data -8 | grad 0 }" shape=record]
    1 [ label = "NodeData { label: \"*\", shape: \"circle\" }" label="*" shape=circle]
    2 [ label = "NodeData { label: \"{ e | data -6 | grad 0 }\", shape: \"record\" }" label="{ e | data -6 | grad 0 }" shape=record]
    3 [ label = "NodeData { label: \"*\", shape: \"circle\" }" label="*" shape=circle]
    4 [ label = "NodeData { label: \"{ b | data -3 | grad 0 }\", shape: \"record\" }" label="{ b | data -3 | grad 0 }" shape=record]
    5 [ label = "NodeData { label: \"{ f | data -2 | grad 0 }\", shape: \"record\" }" label="{ f | data -2 | grad 0 }" shape=record]
    6 [ label = "NodeData { label: \"{ a | data 2 | grad 0 }\", shape: \"record\" }" label="{ a | data 2 | grad 0 }" shape=record]
    7 [ label = "NodeData { label: \"{ d | data 4 | grad 0 }\", shape: \"record\" }" label="{ d | data 4 | grad 0 }" shape=record]
    8 [ label = "NodeData { label: \"+\", shape: \"circle\" }" label="+" shape=circle]
    9 [ label = "NodeData { label: \"{ c | data 10 | grad 0 }\", shape: \"record\" }" label="{ c | data 10 | grad 0 }" shape=record]
    1 -> 0 [ ]
    3 -> 2 [ ]
    8 -> 7 [ ]
    2 -> 8 [ ]
    4 -> 3 [ ]
    5 -> 1 [ ]
    6 -> 3 [ ]
    7 -> 1 [ ]
    9 -> 8 [ ]
}
"#
        );
    }
}
