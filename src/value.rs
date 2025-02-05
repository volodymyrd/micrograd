use petgraph::dot::RankDir::LR;
use petgraph::dot::{Config, Dot};
use petgraph::graph::NodeIndex;
use petgraph::Graph;
use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::{Display, Formatter, Result};
use std::fs::write;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Mul};
use std::process::Command;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct Value {
    data: f64,
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
    let mut op_map = HashMap::new();
    for node in &value_graph.nodes {
        let _node_id = graph.add_node(NodeData::new(
            format!("data {}", node.data),
            "rectangle".to_string(),
        ));
        node_map.insert(Rc::clone(node), _node_id);
        if let Some(op) = node.op {
            let _op_id = graph.add_node(NodeData::new(format!("{}", op), "circle".to_string()));
            graph.add_edge(_op_id, _node_id, ());
            op_map.insert(op, _op_id);
        }
    }
    for (n1, n2) in &value_graph.edges {
        graph.add_edge(node_map[n1], op_map[&n2.op.unwrap()], ());
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
    pub fn new(data: f64) -> Self {
        Self {
            data,
            prev: HashSet::new(),
            op: None,
        }
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
        Self {
            data: self.data + rhs.data,
            prev: vec![Rc::new(self), Rc::new(rhs)].into_iter().collect(),
            op: Some('+'),
        }
    }
}

impl Mul for Value {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            data: self.data * rhs.data,
            prev: vec![Rc::new(self), Rc::new(rhs)].into_iter().collect(),
            op: Some('*'),
        }
    }
}

impl PartialEq for Value {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if self.data.is_nan() {
            return other.data.is_nan();
        }
        if self.data != other.data || self.prev.len() != other.prev.len() || self.op != other.op {
            return false;
        }

        let mut self_prev: Vec<_> = self.prev.iter().collect();
        let mut other_prev: Vec<_> = other.prev.iter().collect();
        self_prev.sort_unstable();
        other_prev.sort_unstable();

        self_prev == other_prev
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
        #[allow(clippy::comparison_chain)]
        if self < other {
            Ordering::Less
        } else if self > other {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}

// canonical raw bit patterns (for hashing)
const CANONICAL_NAN_BITS: u64 = 0x7ff8000000000000u64;

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let bits = if self.data.is_nan() {
            CANONICAL_NAN_BITS
        } else {
            raw_double_bits(&canonicalize_signed_zero(self.data))
        };

        bits.hash(state)
    }
}

#[inline(always)]
fn canonicalize_signed_zero(x: f64) -> f64 {
    // -0.0 + 0.0 == +0.0 under IEEE754 roundTiesToEven rounding mode,
    // which Rust guarantees. Thus by adding a positive zero we
    // canonicalize signed zero without any branches in one instruction.
    x + 0.0
}

// masks for the parts of the IEEE 754 float
const SIGN_MASK: u64 = 0x8000000000000000u64;
const EXP_MASK: u64 = 0x7ff0000000000000u64;
const MAN_MASK: u64 = 0x000fffffffffffffu64;

#[inline]
/// Used for hashing. Input must not be zero or NaN.
fn raw_double_bits(f: &f64) -> u64 {
    let (man, exp, sign) = integer_decode(f);
    let exp_u64 = exp as u16 as u64;
    let sign_u64 = (sign > 0) as u64;
    (man & MAN_MASK) | ((exp_u64 << 52) & EXP_MASK) | ((sign_u64 << 63) & SIGN_MASK)
}

fn integer_decode(f: &f64) -> (u64, i16, i8) {
    let bits: u64 = f.to_bits();
    let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
    let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
    let mantissa = if exponent == 0 {
        (bits & 0xfffffffffffff) << 1
    } else {
        (bits & 0xfffffffffffff) | 0x10000000000000
    };
    // Exponent bias + mantissa shift
    exponent -= 1023 + 52;
    (mantissa, exponent, sign)
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;

    #[test]
    fn test_print_computation_graph() {
        let a = Value::new(2.0);
        let b = Value::new(-3.0);
        let c = Value::new(10.0);
        let d = a * b + c;

        assert_eq!(
            print_computation_graph(Rc::new(d), None),
            r#"digraph {
    rankdir="LR"
    0 [ label = "NodeData { label: \"data -6\", shape: \"rectangle\" }" label="data -6" shape=rectangle]
    1 [ label = "NodeData { label: \"*\", shape: \"circle\" }" label="*" shape=circle]
    2 [ label = "NodeData { label: \"data -3\", shape: \"rectangle\" }" label="data -3" shape=rectangle]
    3 [ label = "NodeData { label: \"data 2\", shape: \"rectangle\" }" label="data 2" shape=rectangle]
    4 [ label = "NodeData { label: \"data 4\", shape: \"rectangle\" }" label="data 4" shape=rectangle]
    5 [ label = "NodeData { label: \"+\", shape: \"circle\" }" label="+" shape=circle]
    6 [ label = "NodeData { label: \"data 10\", shape: \"rectangle\" }" label="data 10" shape=rectangle]
    1 -> 0 [ ]
    5 -> 4 [ ]
    0 -> 5 [ ]
    2 -> 1 [ ]
    3 -> 1 [ ]
    6 -> 5 [ ]
}
"#
        );
    }
}
