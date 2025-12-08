#!/usr/bin/env python3
"""
LEARNING FEEDBACK LOOP
Continuous learning from all system interactions and outcomes
"""

import json
import time
import random
import hashlib
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np

@dataclass
class LearningExperience:
    """Single learning experience"""
    id: str
    context: Dict[str, Any]
    action: str
    outcome: Dict[str, Any]
    success: bool
    learning_points: List[str]
    timestamp: float = field(default_factory=time.time)
    weight: float = 1.0

@dataclass
class LearnedPattern:
    """Pattern learned from experiences"""
    pattern_id: str
    context_pattern: Dict[str, Any]
    action_pattern: str
    expected_outcome: Dict[str, Any]
    confidence: float
    occurrences: int = 1
    last_observed: float = field(default_factory=time.time)

class LearningFeedbackLoop:
    """Continuous learning from feedback"""
    
    def __init__(self, memory_size=10000):
        self.experiences = deque(maxlen=memory_size)
        self.learned_patterns = {}
        self.pattern_confidence = {}
        self.learning_thread = None
        self.running = True
        
        # Learning parameters
        self.learning_rate = 0.1
        self.pattern_threshold = 0.7
        self.experience_decay = 0.99
        
        # Statistics
        self.stats = {
            'experiences_recorded': 0,
            'patterns_learned': 0,
            'success_rate': 0.0,
            'learning_cycles': 0
        }
        
        self._start_learning_loop()
    
    def _start_learning_loop(self):
        """Start continuous learning process"""
        self.learning_thread = threading.Thread(
            target=self._learning_loop,
            daemon=True
        )
        self.learning_thread.start()
        print("🧠 Learning feedback loop started")
    
    def record_experience(self, context: Dict[str, Any], 
                         action: str, 
                         outcome: Dict[str, Any],
                         success: bool,
                         learning_points: Optional[List[str]] = None):
        """Record a learning experience"""
        exp_id = hashlib.md5(
            f"{context}{action}{time.time()}".encode()
        ).hexdigest()[:12]
        
        experience = LearningExperience(
            id=exp_id,
            context=context,
            action=action,
            outcome=outcome,
            success=success,
            learning_points=learning_points or []
        )
        
        self.experiences.append(experience)
        self.stats['experiences_recorded'] += 1
        
        # Update success rate
        total = self.stats['experiences_recorded']
        if success:
            self.stats['success_rate'] = (
                (self.stats['success_rate'] * (total - 1) + 1) / total
            )
        else:
            self.stats['success_rate'] = (
                self.stats['success_rate'] * (total - 1) / total
            )
        
        return exp_id
    
    def _learning_loop(self):
        """Main learning processing loop"""
        while self.running:
            try:
                self.stats['learning_cycles'] += 1
                
                # Process new experiences
                self._process_experiences()
                
                # Extract patterns
                self._extract_patterns()
                
                # Consolidate learning
                self._consolidate_learning()
                
                # Prune old patterns
                self._prune_patterns()
                
                # Sleep between cycles
                time.sleep(10)  # Learn every 10 seconds
                
            except Exception as e:
                print(f"Learning loop error: {e}")
                time.sleep(5)
    
    def _process_experiences(self):
        """Process recent experiences for learning"""
        if not self.experiences:
            return
        
        # Focus on recent experiences
        recent = list(self.experiences)[-100:]  # Last 100
        
        for exp in recent:
            # Decay experience weight
            exp.weight *= self.experience_decay
            
            # Skip if weight too low
            if exp.weight < 0.1:
                continue
            
            # Try to match with existing patterns
            matched = self._match_experience_to_patterns(exp)
            
            if not matched:
                # Could become new pattern
                self._consider_new_pattern(exp)
    
    def _match_experience_to_patterns(self, exp: LearningExperience) -> bool:
        """Try to match experience with existing patterns"""
        best_match = None
        best_similarity = 0.0
        
        for pattern_id, pattern in self.learned_patterns.items():
            similarity = self._calculate_similarity(exp.context, pattern.context_pattern)
            
            if (similarity > self.pattern_threshold and 
                similarity > best_similarity and
                exp.action == pattern.action_pattern):
                
                best_similarity = similarity
                best_match = pattern_id
        
        if best_match:
            # Update pattern with this experience
            pattern = self.learned_patterns[best_match]
            pattern.occurrences += 1
            pattern.last_observed = time.time()
            
            # Update confidence based on outcome match
            outcome_similarity = self._calculate_similarity(
                exp.outcome, pattern.expected_outcome
            )
            
            if exp.success:
                pattern.confidence = pattern.confidence * 0.9 + outcome_similarity * 0.1
            else:
                pattern.confidence = pattern.confidence * 0.95  # Slight decay
            
            return True
        
        return False
    
    def _consider_new_pattern(self, exp: LearningExperience):
        """Consider creating new pattern from experience"""
        # Check if similar context exists
        similar_contexts = []
        
        for other in self.experiences:
            if other.id == exp.id:
                continue
            
            similarity = self._calculate_similarity(exp.context, other.context)
            if (similarity > 0.8 and 
                exp.action == other.action and
                exp.success == other.success):
                
                similar_contexts.append(other)
        
        # Need multiple similar experiences to form pattern
        if len(similar_contexts) >= 2:
            pattern_id = hashlib.md5(
                f"pattern_{exp.action}_{time.time()}".encode()
            ).hexdigest()[:12]
            
            # Create pattern from similar experiences
            pattern = LearnedPattern(
                pattern_id=pattern_id,
                context_pattern=self._average_contexts(
                    [exp.context] + [c.context for c in similar_contexts]
                ),
                action_pattern=exp.action,
                expected_outcome=self._average_outcomes(
                    [exp.outcome] + [c.outcome for c in similar_contexts]
                ),
                confidence=0.7,  # Initial confidence
                occurrences=len(similar_contexts) + 1
            )
            
            self.learned_patterns[pattern_id] = pattern
            self.stats['patterns_learned'] += 1
    
    def _calculate_similarity(self, dict1: Dict, dict2: Dict) -> float:
        """Calculate similarity between two dictionaries"""
        if not dict1 or not dict2:
            return 0.0
        
        keys = set(dict1.keys()) | set(dict2.keys())
        if not keys:
            return 0.0
        
        similarities = []
        
        for key in keys:
            val1 = dict1.get(key)
            val2 = dict2.get(key)
            
            if val1 is None or val2 is None:
                similarities.append(0.0)
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity
                max_val = max(abs(val1), abs(val2), 1.0)
                sim = 1.0 - abs(val1 - val2) / max_val
                similarities.append(max(0.0, sim))
            elif isinstance(val1, str) and isinstance(val2, str):
                # String similarity (simple)
                if val1 == val2:
                    similarities.append(1.0)
                else:
                    # Check for partial matches
                    words1 = set(val1.lower().split())
                    words2 = set(val2.lower().split())
                    if words1 and words2:
                        sim = len(words1 & words2) / len(words1 | words2)
                        similarities.append(sim)
                    else:
                        similarities.append(0.0)
            elif isinstance(val1, bool) and isinstance(val2, bool):
                similarities.append(1.0 if val1 == val2 else 0.0)
            else:
                similarities.append(0.0)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _average_contexts(self, contexts: List[Dict]) -> Dict:
        """Average multiple contexts into one pattern"""
        if not contexts:
            return {}
        
        result = {}
        all_keys = set()
        
        for ctx in contexts:
            all_keys.update(ctx.keys())
        
        for key in all_keys:
            values = [ctx.get(key) for ctx in contexts if key in ctx]
            
            if not values:
                continue
            
            # Handle different types
            first_val = values[0]
            
            if isinstance(first_val, (int, float)):
                # Numeric average
                result[key] = sum(values) / len(values)
            elif isinstance(first_val, bool):
                # Majority boolean
                result[key] = sum(values) > len(values) / 2
            elif isinstance(first_val, str):
                # Most common string
                from collections import Counter
                result[key] = Counter(values).most_common(1)[0][0]
            else:
                # Keep first value
                result[key] = first_val
        
        return result
    
    def _average_outcomes(self, outcomes: List[Dict]) -> Dict:
        """Average multiple outcomes"""
        return self._average_contexts(outcomes)
    
    def _extract_patterns(self):
        """Extract higher-level patterns from learned patterns"""
        # Group patterns by action
        patterns_by_action = defaultdict(list)
        
        for pattern_id, pattern in self.learned_patterns.items():
            patterns_by_action[pattern.action_pattern].append(pattern)
        
        # Look for meta-patterns across actions
        for action, patterns in patterns_by_action.items():
            if len(patterns) < 3:
                continue
            
            # Check for timing patterns
            self._extract_timing_patterns(patterns, action)
            
            # Check for context evolution patterns
            self._extract_evolution_patterns(patterns, action)
    
    def _extract_timing_patterns(self, patterns: List[LearnedPattern], action: str):
        """Extract timing-related patterns"""
        # Analyze when this action is most successful
        success_times = []
        failure_times = []
        
        for pattern in patterns:
            # Convert last_observed to hour of day
            dt = datetime.fromtimestamp(pattern.last_observed)
            hour = dt.hour + dt.minute / 60.0
            
            if pattern.confidence > 0.7:
                success_times.append(hour)
            else:
                failure_times.append(hour)
        
        if success_times and len(success_times) > 5:
            # Found timing pattern
            avg_success_hour = sum(success_times) / len(success_times)
            print(f"⏰ Timing pattern for {action}: "
                  f"Most successful around {avg_success_hour:.1f} hours")
    
    def _extract_evolution_patterns(self, patterns: List[LearnedPattern], action: str):
        """Extract how patterns evolve over time"""
        # Sort by last observed
        sorted_patterns = sorted(patterns, key=lambda p: p.last_observed)
        
        if len(sorted_patterns) < 4:
            return
        
        # Check for confidence trends
        confidences = [p.confidence for p in sorted_patterns]
        
        # Simple trend detection
        if len(confidences) >= 4:
            first_half = confidences[:len(confidences)//2]
            second_half = confidences[len(confidences)//2:]
            
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)
            
            if avg_second > avg_first * 1.1:
                print(f"📈 Learning improving for {action}: "
                      f"{avg_first:.2f} → {avg_second:.2f}")
            elif avg_second < avg_first * 0.9:
                print(f"📉 Learning declining for {action}: "
                      f"{avg_first:.2f} → {avg_second:.2f}")
    
    def _consolidate_learning(self):
        """Consolidate and generalize learning"""
        # Merge similar patterns
        to_merge = []
        pattern_ids = list(self.learned_patterns.keys())
        
        for i in range(len(pattern_ids)):
            for j in range(i + 1, len(pattern_ids)):
                id1, id2 = pattern_ids[i], pattern_ids[j]
                p1, p2 = self.learned_patterns[id1], self.learned_patterns[id2]
                
                if (p1.action_pattern == p2.action_pattern and
                    self._calculate_similarity(p1.context_pattern, p2.context_pattern) > 0.9):
                    
                    to_merge.append((id1, id2))
        
        # Merge patterns
        merged = set()
        for id1, id2 in to_merge:
            if id1 in merged or id2 in merged:
                continue
            
            p1, p2 = self.learned_patterns[id1], self.learned_patterns[id2]
            
            # Merge into p1
            p1.context_pattern = self._average_contexts(
                [p1.context_pattern, p2.context_pattern]
            )
            p1.expected_outcome = self._average_outcomes(
                [p1.expected_outcome, p2.expected_outcome]
            )
            p1.occurrences += p2.occurrences
            p1.confidence = (p1.confidence * p1.occurrences + 
                           p2.confidence * p2.occurrences) / (p1.occurrences + p2.occurrences)
            
            # Mark p2 for removal
            merged.add(id2)
        
        # Remove merged patterns
        for pattern_id in merged:
            if pattern_id in self.learned_patterns:
                del self.learned_patterns[pattern_id]
    
    def _prune_patterns(self):
        """Prune old or low-confidence patterns"""
        current_time = time.time()
        to_remove = []
        
        for pattern_id, pattern in self.learned_patterns.items():
            # Remove very old patterns (30 days)
            if current_time - pattern.last_observed > 30 * 24 * 3600:
                to_remove.append(pattern_id)
            # Remove very low confidence patterns
            elif pattern.confidence < 0.3 and pattern.occurrences < 3:
                to_remove.append(pattern_id)
        
        for pattern_id in to_remove:
            del self.learned_patterns[pattern_id]
    
    def get_recommendation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get action recommendation based on learned patterns"""
        if not self.learned_patterns:
            return {"action": "explore", "confidence": 0.0}
        
        best_pattern = None
        best_score = 0.0
        
        for pattern_id, pattern in self.learned_patterns.items():
            similarity = self._calculate_similarity(context, pattern.context_pattern)
            score = similarity * pattern.confidence
            
            if score > best_score:
                best_score = score
                best_pattern = pattern
        
        if best_pattern and best_score > 0.5:
            return {
                "action": best_pattern.action_pattern,
                "confidence": best_score,
                "expected_outcome": best_pattern.expected_outcome,
                "based_on_pattern": best_pattern.pattern_id
            }
        else:
            # No good match - suggest exploration
            return {
                "action": "explore",
                "confidence": 0.0,
                "message": "No strong pattern match - exploring"
            }
    
    def get_learning_report(self) -> Dict[str, Any]:
        """Get learning status report"""
        return {
            "timestamp": time.time(),
            "statistics": self.stats.copy(),
            "patterns": {
                "total": len(self.learned_patterns),
                "by_confidence": {
                    "high": sum(1 for p in self.learned_patterns.values() 
                              if p.confidence > 0.8),
                    "medium": sum(1 for p in self.learned_patterns.values() 
                                if 0.5 <= p.confidence <= 0.8),
                    "low": sum(1 for p in self.learned_patterns.values() 
                             if p.confidence < 0.5),
                }
            },
            "recent_experiences": min(100, len(self.experiences)),
            "success_rate": self.stats['success_rate']
        }
    
    def stop(self):
        """Stop learning loop"""
        self.running = False
        if self.learning_thread:
            self.learning_thread.join(timeout=5)

# Global instance
learner = LearningFeedbackLoop()

def learn_from_experience(context: Dict, action: str, outcome: Dict, 
                         success: bool, notes: List[str] = None):
    """Quick function to record learning"""
    return learner.record_experience(context, action, outcome, success, notes)

def get_learned_action(context: Dict) -> str:
    """Get recommended action based on learning"""
    recommendation = learner.get_recommendation(context)
    return recommendation['action']

if __name__ == "__main__":
    # Example learning
    print("🧪 Testing learning feedback loop...")
    
    # Record some example experiences
    for i in range(5):
        context = {"user_type": "admin", "time_of_day": i % 24}
        action = "grant_access" if i % 2 == 0 else "deny_access"
        outcome = {"success": i % 2 == 0, "access_granted": i % 2 == 0}
        
        learn_from_experience(
            context=context,
            action=action,
            outcome=outcome,
            success=i % 2 == 0,
            notes=[f"Test experience {i}"]
        )
    
    # Get learning report
    report = learner.get_learning_report
