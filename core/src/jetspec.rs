use std::collections::{BTreeMap, BTreeSet};

#[derive(uniffi::Record, Clone, Debug, PartialEq)]
pub struct DraftCandidateNode {
    pub node_id: u32,
    pub parent_id: u32,
    pub token_id: u32,
    pub depth: u32,
    pub branch_idx: u32,
    pub score: f32,
}

#[derive(uniffi::Record, Clone, Debug)]
pub struct JetSpecTreeConfig {
    pub max_depth: u32,
    pub branching_factor: u32,
    pub top_k: u32,
    pub max_nodes: u32,
}

impl Default for JetSpecTreeConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            branching_factor: 2,
            top_k: 8,
            max_nodes: 7, // 1 root (virtual) + 2 (d1) + 4 (d2) = 7
        }
    }
}

#[derive(uniffi::Record, Clone, Debug)]
pub struct JetSpecTreeMask {
    pub node_count: u32,
    pub mask: Vec<f32>, // Flattened node_count x node_count (0.0 for ancestor, -1e9 for non-ancestor)
    pub parent_indices: Vec<u32>,
    pub depths: Vec<u32>,
    pub token_ids: Vec<u32>,
}

#[derive(uniffi::Record, Clone, Debug, PartialEq)]
pub struct JetSpecAcceptedResult {
    pub accepted_tokens: Vec<u32>,
    pub accepted_node_indices: Vec<u32>,
    pub bonus_token: Option<u32>,
    pub accepted_count: u32,
    pub effective_tau: f32,
}

#[derive(Clone, Debug)]
pub struct DraftTreeTopology {
    pub nodes: Vec<DraftCandidateNode>,
    pub children_map: BTreeMap<u32, Vec<u32>>,
}

impl DraftTreeTopology {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            children_map: BTreeMap::new(),
        }
    }

    /// Construct a candidate tree from flat draft tokens and scores.
    /// Each depth level d has (branching_factor^d) nodes, or user-supplied candidate structure.
    pub fn from_candidate_tokens(
        root_token_id: u32,
        draft_tokens: &[u32],
        draft_scores: &[f32],
        depth: u32,
        branching_factor: u32,
        max_nodes: u32,
    ) -> Self {
        let mut tree = Self::new();

        // Node 0 is the root node (the last accepted prompt / target token)
        let root_node = DraftCandidateNode {
            node_id: 0,
            parent_id: 0,
            token_id: root_token_id,
            depth: 0,
            branch_idx: 0,
            score: 1.0,
        };
        tree.nodes.push(root_node);

        if draft_tokens.is_empty() || depth == 0 || branching_factor == 0 {
            return tree;
        }

        let mut token_idx = 0usize;
        let mut current_parents = vec![0u32];

        for d in 1..=depth {
            let mut next_parents = Vec::new();
            for &p_id in &current_parents {
                for b in 0..branching_factor {
                    if tree.nodes.len() as u32 >= max_nodes || token_idx >= draft_tokens.len() {
                        break;
                    }
                    let node_id = tree.nodes.len() as u32;
                    let tok = draft_tokens[token_idx];
                    let score = if token_idx < draft_scores.len() {
                        draft_scores[token_idx]
                    } else {
                        0.0
                    };
                    token_idx += 1;

                    let node = DraftCandidateNode {
                        node_id,
                        parent_id: p_id,
                        token_id: tok,
                        depth: d,
                        branch_idx: b,
                        score,
                    };
                    tree.nodes.push(node);
                    tree.children_map.entry(p_id).or_default().push(node_id);
                    next_parents.push(node_id);
                }
            }
            current_parents = next_parents;
            if current_parents.is_empty() {
                break;
            }
        }

        tree
    }

    /// Returns the sequence of node IDs from root (0) to target node (inclusive)
    pub fn get_ancestor_path(&self, node_id: u32) -> Vec<u32> {
        let mut path = Vec::new();
        let mut curr = node_id as usize;
        while curr < self.nodes.len() {
            path.push(curr as u32);
            if curr == 0 {
                break;
            }
            curr = self.nodes[curr].parent_id as usize;
        }
        path.reverse();
        path
    }

    /// Check if ancestor_id is an ancestor of node_id (or same node)
    pub fn is_ancestor(&self, ancestor_id: u32, node_id: u32) -> bool {
        if ancestor_id == node_id {
            return true;
        }
        let mut curr = node_id as usize;
        while curr != 0 && curr < self.nodes.len() {
            curr = self.nodes[curr].parent_id as usize;
            if curr as u32 == ancestor_id {
                return true;
            }
        }
        false
    }

    /// Generates flattened N x N tree-causal attention mask where:
    /// mask[i * N + j] = 0.0 if j is ancestor of i (or j == i), else -1e9
    pub fn generate_tree_mask(&self) -> JetSpecTreeMask {
        let n = self.nodes.len();
        let mut mask = vec![-1.0e9f32; n * n];
        let mut parent_indices = Vec::with_capacity(n);
        let mut depths = Vec::with_capacity(n);
        let mut token_ids = Vec::with_capacity(n);

        for (i, node_i) in self.nodes.iter().enumerate() {
            parent_indices.push(node_i.parent_id);
            depths.push(node_i.depth);
            token_ids.push(node_i.token_id);

            // Traverse ancestors of i
            let path = self.get_ancestor_path(i as u32);
            for &j in &path {
                mask[i * n + (j as usize)] = 0.0;
            }
        }

        JetSpecTreeMask {
            node_count: n as u32,
            mask,
            parent_indices,
            depths,
            token_ids,
        }
    }

    /// Dynamic Tree Pruning for NVMe SSD MoE streaming:
    /// Prunes leaf nodes with lowest confidence if total unique active experts exceeds budget.
    pub fn prune_for_expert_budget(
        &self,
        candidate_experts_per_node: &[Vec<u32>],
        max_unique_experts: usize,
    ) -> Self {
        if self.nodes.is_empty() || candidate_experts_per_node.len() != self.nodes.len() {
            return self.clone();
        }

        let mut active_nodes: BTreeSet<u32> = (0..self.nodes.len() as u32).collect();

        loop {
            // Compute union of active experts
            let mut unique_experts = BTreeSet::new();
            for &n_id in &active_nodes {
                let idx = n_id as usize;
                if idx < candidate_experts_per_node.len() {
                    for &exp in &candidate_experts_per_node[idx] {
                        unique_experts.insert(exp);
                    }
                }
            }

            if unique_experts.len() <= max_unique_experts || active_nodes.len() <= 1 {
                break;
            }

            // Find active leaf nodes (nodes with no active children)
            let mut leaf_nodes = Vec::new();
            for &n_id in &active_nodes {
                if n_id == 0 { continue; } // Never prune root
                let has_active_child = self.children_map.get(&n_id).map_or(false, |children| {
                    children.iter().any(|c| active_nodes.contains(c))
                });
                if !has_active_child {
                    leaf_nodes.push(n_id);
                }
            }

            if leaf_nodes.is_empty() {
                break;
            }

            // Prune the leaf node with the lowest score
            leaf_nodes.sort_by(|&a, &b| {
                let score_a = self.nodes[a as usize].score;
                let score_b = self.nodes[b as usize].score;
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            });

            let worst_leaf = leaf_nodes[0];
            active_nodes.remove(&worst_leaf);
        }

        // Rebuild compacted tree with active nodes
        let mut old_to_new = BTreeMap::new();
        let mut new_tree = Self::new();

        for old_id in 0..self.nodes.len() as u32 {
            if active_nodes.contains(&old_id) {
                let new_id = new_tree.nodes.len() as u32;
                old_to_new.insert(old_id, new_id);

                let old_node = &self.nodes[old_id as usize];
                let new_parent_id = if old_id == 0 {
                    0
                } else {
                    *old_to_new.get(&old_node.parent_id).unwrap_or(&0)
                };

                let new_node = DraftCandidateNode {
                    node_id: new_id,
                    parent_id: new_parent_id,
                    token_id: old_node.token_id,
                    depth: old_node.depth,
                    branch_idx: old_node.branch_idx,
                    score: old_node.score,
                };
                new_tree.nodes.push(new_node);
                if new_id > 0 {
                    new_tree.children_map.entry(new_parent_id).or_default().push(new_id);
                }
            }
        }

        new_tree
    }
}

pub struct JetSpecAcceptanceOracle;

impl JetSpecAcceptanceOracle {
    /// Greedy verification oracle:
    /// Traverses the candidate tree top-down. At each node p, compares the draft token with
    /// argmax(target_logits[parent(p)]). Accepts the matching branch and samples a bonus token.
    pub fn verify_greedy(
        tree: &DraftTreeTopology,
        target_logits: &[f32],
        vocab_size: usize,
    ) -> JetSpecAcceptedResult {
        if tree.nodes.is_empty() || target_logits.is_empty() || vocab_size == 0 {
            return JetSpecAcceptedResult {
                accepted_tokens: Vec::new(),
                accepted_node_indices: Vec::new(),
                bonus_token: None,
                accepted_count: 0,
                effective_tau: 0.0,
            };
        }

        let num_nodes = tree.nodes.len();
        let expected_logits_len = num_nodes * vocab_size;
        if target_logits.len() < expected_logits_len {
            // Insufficient logits passed
            return JetSpecAcceptedResult {
                accepted_tokens: Vec::new(),
                accepted_node_indices: Vec::new(),
                bonus_token: None,
                accepted_count: 0,
                effective_tau: 0.0,
            };
        }

        let mut accepted_tokens = Vec::new();
        let mut accepted_node_indices = Vec::new();
        let mut curr_node_id = 0u32;

        loop {
            // Find target argmax token at curr_node_id position
            let row_start = (curr_node_id as usize) * vocab_size;
            let row_logits = &target_logits[row_start..row_start + vocab_size];

            let mut best_val = f32::NEG_INFINITY;
            let mut target_argmax_token = 0u32;
            for (v_idx, &val) in row_logits.iter().enumerate() {
                if val > best_val {
                    best_val = val;
                    target_argmax_token = v_idx as u32;
                }
            }

            // Check if any child of curr_node_id matches target_argmax_token
            let children = tree.children_map.get(&curr_node_id);
            let mut matched_child = None;

            if let Some(children_list) = children {
                for &child_id in children_list {
                    if tree.nodes[child_id as usize].token_id == target_argmax_token {
                        matched_child = Some(child_id);
                        break;
                    }
                }
            }

            if let Some(child_id) = matched_child {
                // Draft token accepted!
                accepted_tokens.push(target_argmax_token);
                accepted_node_indices.push(child_id);
                curr_node_id = child_id;
            } else {
                // Mismatch or end of tree: bonus/corrective token is target_argmax_token
                let accepted_count = accepted_tokens.len() as u32;
                let effective_tau = (accepted_count + 1) as f32;
                return JetSpecAcceptedResult {
                    accepted_tokens,
                    accepted_node_indices,
                    bonus_token: Some(target_argmax_token),
                    accepted_count,
                    effective_tau,
                };
            }
        }
    }

    /// Speculative sampling verification oracle with temperature and top-p support.
    pub fn verify_speculative_sampling(
        tree: &DraftTreeTopology,
        draft_probs: &[f32],
        target_logits: &[f32],
        vocab_size: usize,
        temperature: f32,
        rng_seed: u64,
    ) -> JetSpecAcceptedResult {
        let mut rng_state = rng_seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut rand_f32 = || -> f32 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 32) as u32 as f32) / (u32::MAX as f32)
        };

        let temp = if temperature <= 0.01 { 1.0 } else { temperature };
        let mut accepted_tokens = Vec::new();
        let mut accepted_node_indices = Vec::new();
        let mut curr_node_id = 0u32;

        loop {
            let row_start = (curr_node_id as usize) * vocab_size;
            if row_start + vocab_size > target_logits.len() {
                break;
            }
            let row_logits = &target_logits[row_start..row_start + vocab_size];

            // Compute target softmax with temperature
            let mut max_l = f32::NEG_INFINITY;
            for &l in row_logits {
                if l > max_l { max_l = l; }
            }
            let mut sum_exp = 0.0f32;
            let mut p_target = vec![0.0f32; vocab_size];
            for (i, &l) in row_logits.iter().enumerate() {
                let exp_val = ((l - max_l) / temp).exp();
                p_target[i] = exp_val;
                sum_exp += exp_val;
            }
            let inv_sum = 1.0 / (sum_exp + 1e-12);
            for p in &mut p_target {
                *p *= inv_sum;
            }

            // Inspect children of curr_node_id
            let children = tree.children_map.get(&curr_node_id);
            let mut accepted_child = None;

            if let Some(children_list) = children {
                for &child_id in children_list {
                    let draft_tok = tree.nodes[child_id as usize].token_id;
                    let target_prob = p_target[draft_tok as usize];
                    let draft_prob_val = if !draft_probs.is_empty() && (child_id as usize) * vocab_size + (draft_tok as usize) < draft_probs.len() {
                        draft_probs[(child_id as usize) * vocab_size + (draft_tok as usize)]
                    } else {
                        tree.nodes[child_id as usize].score.max(1e-4)
                    };

                    let accept_prob = (target_prob / draft_prob_val).min(1.0);
                    let r = rand_f32();
                    if r <= accept_prob {
                        accepted_child = Some((child_id, draft_tok));
                        break;
                    }
                }
            }

            if let Some((child_id, tok)) = accepted_child {
                accepted_tokens.push(tok);
                accepted_node_indices.push(child_id);
                curr_node_id = child_id;
            } else {
                // Rejection: sample bonus token from target distribution
                let r = rand_f32();
                let mut cum = 0.0f32;
                let mut bonus = 0u32;
                for (v_idx, &p) in p_target.iter().enumerate() {
                    cum += p;
                    if r <= cum || v_idx == vocab_size - 1 {
                        bonus = v_idx as u32;
                        break;
                    }
                }

                let accepted_count = accepted_tokens.len() as u32;
                let effective_tau = (accepted_count + 1) as f32;
                return JetSpecAcceptedResult {
                    accepted_tokens,
                    accepted_node_indices,
                    bonus_token: Some(bonus),
                    accepted_count,
                    effective_tau,
                };
            }
        }

        let accepted_count = accepted_tokens.len() as u32;
        JetSpecAcceptedResult {
            accepted_tokens,
            accepted_node_indices,
            bonus_token: None,
            accepted_count,
            effective_tau: accepted_count as f32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_construction_and_mask() {
        // Root token: 100
        // Draft tokens: [101, 102, 103, 104, 105, 106]
        // Depth 2, branching factor 2 -> 1 root + 2 depth1 + 4 depth2 = 7 nodes
        let draft_tokens = vec![101, 102, 103, 104, 105, 106];
        let draft_scores = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4];
        let tree = DraftTreeTopology::from_candidate_tokens(100, &draft_tokens, &draft_scores, 2, 2, 7);

        assert_eq!(tree.nodes.len(), 7);
        assert_eq!(tree.nodes[0].token_id, 100);
        assert_eq!(tree.nodes[0].depth, 0);

        // Depth 1 children of 0: nodes 1 (101) and 2 (102)
        assert_eq!(tree.nodes[1].parent_id, 0);
        assert_eq!(tree.nodes[1].depth, 1);
        assert_eq!(tree.nodes[1].token_id, 101);

        assert_eq!(tree.nodes[2].parent_id, 0);
        assert_eq!(tree.nodes[2].depth, 1);
        assert_eq!(tree.nodes[2].token_id, 102);

        // Depth 2 children of 1: nodes 3 (103) and 4 (104)
        assert_eq!(tree.nodes[3].parent_id, 1);
        assert_eq!(tree.nodes[3].depth, 2);
        assert_eq!(tree.nodes[4].parent_id, 1);
        assert_eq!(tree.nodes[4].depth, 2);

        // Ancestor paths
        assert_eq!(tree.get_ancestor_path(0), vec![0]);
        assert_eq!(tree.get_ancestor_path(1), vec![0, 1]);
        assert_eq!(tree.get_ancestor_path(3), vec![0, 1, 3]);
        assert_eq!(tree.get_ancestor_path(5), vec![0, 2, 5]);

        // Generate mask
        let tree_mask = tree.generate_tree_mask();
        assert_eq!(tree_mask.node_count, 7);
        let n = 7;

        // Node 3 (ancestors: 0, 1, 3)
        assert_eq!(tree_mask.mask[3 * n + 0], 0.0);
        assert_eq!(tree_mask.mask[3 * n + 1], 0.0);
        assert_eq!(tree_mask.mask[3 * n + 2], -1.0e9);
        assert_eq!(tree_mask.mask[3 * n + 3], 0.0);
        assert_eq!(tree_mask.mask[3 * n + 4], -1.0e9);
    }

    #[test]
    fn test_greedy_acceptance_oracle() {
        let draft_tokens = vec![101, 102, 103, 104, 105, 106];
        let draft_scores = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4];
        let tree = DraftTreeTopology::from_candidate_tokens(100, &draft_tokens, &draft_scores, 2, 2, 7);

        let vocab_size = 200;
        let mut target_logits = vec![0.0f32; 7 * vocab_size];

        // Step 0 (root node 0): Target predicts token 101 as best (matches node 1)
        target_logits[0 * vocab_size + 101] = 10.0;

        // Step 1 (node 1): Target predicts token 104 as best (matches node 4)
        target_logits[1 * vocab_size + 104] = 10.0;

        // Step 2 (node 4): Target predicts token 150 (bonus token)
        target_logits[4 * vocab_size + 150] = 10.0;

        let result = JetSpecAcceptanceOracle::verify_greedy(&tree, &target_logits, vocab_size);
        assert_eq!(result.accepted_tokens, vec![101, 104]);
        assert_eq!(result.accepted_node_indices, vec![1, 4]);
        assert_eq!(result.bonus_token, Some(150));
        assert_eq!(result.accepted_count, 2);
        assert_eq!(result.effective_tau, 3.0);
    }

    #[test]
    fn test_dynamic_tree_pruning_moe() {
        let draft_tokens = vec![101, 102, 103, 104, 105, 106];
        let draft_scores = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.1];
        let tree = DraftTreeTopology::from_candidate_tokens(100, &draft_tokens, &draft_scores, 2, 2, 7);

        let candidate_experts = vec![
            vec![1, 2],
            vec![3, 4],
            vec![5, 6],
            vec![7, 8],
            vec![9, 10],
            vec![11, 12],
            vec![13, 14],
        ];

        // Cap to 6 unique experts
        let pruned = tree.prune_for_expert_budget(&candidate_experts, 6);
        assert!(pruned.nodes.len() < 7);
        assert!(pruned.nodes.len() >= 1);
        let pruned_mask = pruned.generate_tree_mask();
        assert_eq!(pruned_mask.node_count, pruned.nodes.len() as u32);
    }
}
