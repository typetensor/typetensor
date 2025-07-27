/**
 * Runtime tests for the symbolic shape system
 *
 * These tests validate the actual runtime behavior of symbolic dimensions,
 * constraint solving, and dynamic shape resolution.
 */

import { describe, it, expect, beforeEach } from 'bun:test';
import {
  createSymbolicDim,
  isSymbolicDim,
  getSymbolicName,
  isSameSymbolicDim,
  createConstraint,
  validateConstraint,
  SymbolicEnvironment,
  resolveSymbolicShape,
  CommonSymbols,
  createMLEnvironment,
  LayerShapeResolver,
  canBroadcastSymbolic,
  inferFromConcrete,
  type SymbolicConstraint,
} from './symbolic';
import type { SymbolicShape } from './types';

// =============================================================================
// Symbolic Dimension Creation and Utilities
// =============================================================================

describe('Symbolic Dimension Creation', () => {
  describe('createSymbolicDim', () => {
    it('should create symbolic dimensions with correct names', () => {
      const batch = createSymbolicDim('batch');
      const seqLen = createSymbolicDim('seq_len');

      expect(getSymbolicName(batch)).toBe('batch');
      expect(getSymbolicName(seqLen)).toBe('seq_len');
    });

    it('should create unique instances for same names', () => {
      const batch1 = createSymbolicDim('batch');
      const batch2 = createSymbolicDim('batch');

      // Same name but different instances (object identity)
      expect(getSymbolicName(batch1)).toBe(getSymbolicName(batch2));
      expect(batch1).not.toBe(batch2);
    });
  });

  describe('Symbolic dimension utilities', () => {
    it('should identify symbolic dimensions correctly', () => {
      const batch = createSymbolicDim('batch');

      expect(isSymbolicDim(batch)).toBe(true);
      expect(isSymbolicDim(42)).toBe(false);
      expect(isSymbolicDim('batch')).toBe(false);
      expect(isSymbolicDim({})).toBe(false);
    });

    it('should extract symbolic names', () => {
      const features = createSymbolicDim('features');
      expect(getSymbolicName(features)).toBe('features');
    });

    it('should compare symbolic dimensions by name', () => {
      const batch1 = createSymbolicDim('batch');
      const batch2 = createSymbolicDim('batch');
      const features = createSymbolicDim('features');

      expect(isSameSymbolicDim(batch1, batch1)).toBe(true);
      expect(isSameSymbolicDim(batch1, batch2)).toBe(true); // Same name
      expect(isSameSymbolicDim(batch1, features)).toBe(false);
    });
  });

  describe('CommonSymbols', () => {
    it('should create common ML symbolic dimensions', () => {
      expect(getSymbolicName(CommonSymbols.batch())).toBe('batch');
      expect(getSymbolicName(CommonSymbols.seqLen())).toBe('seq_len');
      expect(getSymbolicName(CommonSymbols.features())).toBe('features');
      expect(getSymbolicName(CommonSymbols.height())).toBe('height');
      expect(getSymbolicName(CommonSymbols.width())).toBe('width');
      expect(getSymbolicName(CommonSymbols.channels())).toBe('channels');
    });

    it('should create custom named dimensions', () => {
      const customBatch = CommonSymbols.batch('custom_batch');
      expect(getSymbolicName(customBatch)).toBe('custom_batch');
    });
  });
});

// =============================================================================
// Constraint Management
// =============================================================================

describe('Constraint Management', () => {
  describe('createConstraint', () => {
    it('should create constraints between symbolic dimensions', () => {
      const batch = createSymbolicDim('batch');
      const features = createSymbolicDim('features');

      const constraint = createConstraint(batch, 'eq', features);

      expect(constraint.left).toBe(batch);
      expect(constraint.right).toBe(features);
      expect(constraint.type).toBe('eq');
      expect(constraint.id).toBeDefined();
    });

    it('should create constraints with numbers', () => {
      const batch = createSymbolicDim('batch');
      const constraint = createConstraint(batch, 'gt', 0);

      expect(constraint.left).toBe(batch);
      expect(constraint.right).toBe(0);
      expect(constraint.type).toBe('gt');
    });

    it('should include descriptions when provided', () => {
      const batch = createSymbolicDim('batch');
      const constraint = createConstraint(batch, 'gte', 1, 'Batch size must be positive');

      expect(constraint.description).toBe('Batch size must be positive');
    });

    it('should generate unique constraint IDs', () => {
      const batch = createSymbolicDim('batch');
      const constraint1 = createConstraint(batch, 'eq', 32);
      const constraint2 = createConstraint(batch, 'eq', 32);

      expect(constraint1.id).not.toBe(constraint2.id);
    });
  });

  describe('validateConstraint', () => {
    it('should validate numeric constraints', () => {
      const validConstraint: SymbolicConstraint = {
        id: 'test',
        left: 10,
        right: 5,
        type: 'gt',
      };

      expect(validateConstraint(validConstraint)).toBe(true);

      const invalidConstraint: SymbolicConstraint = {
        id: 'test',
        left: 5,
        right: 10,
        type: 'gt',
      };

      expect(validateConstraint(invalidConstraint)).toBe(false);
    });

    it('should validate symbolic constraints as always valid to define', () => {
      const batch = createSymbolicDim('batch');
      const features = createSymbolicDim('features');

      const constraint = createConstraint(batch, 'eq', features);
      expect(validateConstraint(constraint)).toBe(true);
    });
  });
});

// =============================================================================
// SymbolicEnvironment
// =============================================================================

describe('SymbolicEnvironment', () => {
  let env: SymbolicEnvironment;

  beforeEach(() => {
    env = new SymbolicEnvironment();
  });

  describe('Dimension Definition and Binding', () => {
    it('should define symbolic dimensions', () => {
      const batch = env.define('batch');

      expect(isSymbolicDim(batch)).toBe(true);
      expect(getSymbolicName(batch)).toBe('batch');
    });

    it('should define and bind dimensions in one step', () => {
      const batch = env.define('batch', 32);

      expect(env.getBind(batch)).toBe(32);
    });

    it('should bind dimensions to values', () => {
      const batch = env.define('batch');
      env.bind(batch, 16);

      expect(env.getBind(batch)).toBe(16);
    });

    it('should validate binding values', () => {
      const batch = env.define('batch');

      expect(() => {
        env.bind(batch, -1);
      }).toThrow('negative value');
      expect(() => {
        env.bind(batch, 2.5);
      }).toThrow('non-integer value');
    });

    it('should get bindings for unbound dimensions', () => {
      const batch = env.define('batch');

      expect(env.getBind(batch)).toBeUndefined();
      expect(env.isBound(batch)).toBe(false);
    });
  });

  describe('Constraint Management', () => {
    it('should add constraints', () => {
      const batch = env.define('batch');
      const minBatch = env.define('min_batch', 1);

      env.constrain(batch, 'gte', minBatch, 'Batch must be at least min_batch');

      const constraints = env.getConstraints();
      expect(constraints).toHaveLength(1);
      expect(constraints[0]?.type).toBe('gte');
    });

    it('should validate numeric constraints before adding', () => {
      expect(() => {
        env.constrain(5, 'gt', 10);
      }).toThrow('Invalid constraint');
    });

    it('should provide equality constraint convenience method', () => {
      const batch = env.define('batch');
      const features = env.define('features');

      env.equal(batch, features);

      const constraints = env.getConstraints();
      expect(constraints[0]?.type).toBe('eq');
    });
  });

  describe('Environment Management', () => {
    it('should get all bindings', () => {
      env.define('batch', 32);
      env.define('features', 768);

      const bindings = env.getBindings();
      expect(bindings.size).toBe(2);
      expect(bindings.get('batch')).toBe(32);
      expect(bindings.get('features')).toBe(768);
    });

    it('should clear all data', () => {
      env.define('batch', 32);
      env.clear();

      expect(env.getBindings().size).toBe(0);
      expect(env.getConstraints()).toHaveLength(0);
    });

    it('should clone environments', () => {
      const batch = env.define('batch', 32);
      const cloned = env.clone();

      expect(cloned.getBind(batch)).toBe(32);

      // Changes to clone shouldn't affect original
      cloned.bind(batch, 64);
      expect(env.getBind(batch)).toBe(32);
      expect(cloned.getBind(batch)).toBe(64);
    });
  });

  describe('createMLEnvironment', () => {
    it('should create environment with common ML dimensions', () => {
      const mlEnv = createMLEnvironment();

      // These should be defined but unbound
      const batch = mlEnv.define('batch'); // Won't re-define if already exists
      const seqLen = mlEnv.define('seq_len');

      expect(isSymbolicDim(batch)).toBe(true);
      expect(isSymbolicDim(seqLen)).toBe(true);
      expect(mlEnv.isBound(batch)).toBe(false);
      expect(mlEnv.isBound(seqLen)).toBe(false);
    });
  });
});

// =============================================================================
// Symbolic Shape Resolution
// =============================================================================

describe('Symbolic Shape Resolution', () => {
  describe('resolveSymbolicShape', () => {
    it('should resolve fully bound shapes', () => {
      const env = new SymbolicEnvironment();
      const batch = env.define('batch', 32);
      const features = env.define('features', 768);

      const symbolicShape: SymbolicShape = [batch, features];
      const result = resolveSymbolicShape(symbolicShape, env);

      expect(result.success).toBe(true);
      if (result.success && result.type === 'resolved') {
        expect(Array.from(result.shape)).toEqual([32, 768]);
      }
    });

    it('should resolve mixed numeric and symbolic shapes', () => {
      const env = new SymbolicEnvironment();
      const batch = env.define('batch', 16);

      const symbolicShape: SymbolicShape = [batch, 128, 768];
      const result = resolveSymbolicShape(symbolicShape, env);

      expect(result.success).toBe(true);
      if (result.success && result.type === 'resolved') {
        expect(Array.from(result.shape)).toEqual([16, 128, 768]);
      }
    });

    it('should handle unbound dimensions in non-strict mode', () => {
      const env = new SymbolicEnvironment();
      const batch = env.define('batch'); // Not bound

      const symbolicShape: SymbolicShape = [batch, 768];
      const result = resolveSymbolicShape(symbolicShape, env, { strict: false });

      // In non-strict mode, returns partial shape with warnings
      expect(result.success).toBe(true);
      if (result.success && result.type === 'partial') {
        expect(result.warnings).toContain('Unbound symbolic dimension: batch');
      }
    });

    it('should fail on unbound dimensions in strict mode', () => {
      const env = new SymbolicEnvironment();
      const batch = env.define('batch'); // Not bound

      const symbolicShape: SymbolicShape = [batch, 768];
      const result = resolveSymbolicShape(symbolicShape, env, { strict: true });

      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.errors[0]).toContain('Cannot resolve symbolic dimensions');
      }
    });

    it('should infer dimensions from equality constraints', () => {
      const env = new SymbolicEnvironment();
      const batch = env.define('batch');
      const seqLen = env.define('seq_len', 128);

      env.equal(batch, seqLen);

      const symbolicShape: SymbolicShape = [batch, 768];
      const result = resolveSymbolicShape(symbolicShape, env);

      expect(result.success).toBe(true);
      if (result.success && result.type === 'resolved') {
        expect(Array.from(result.shape)).toEqual([128, 768]);
      }
    });

    it('should infer from numeric equality constraints', () => {
      const env = new SymbolicEnvironment();
      const batch = env.define('batch');

      env.constrain(batch, 'eq', 64);

      const symbolicShape: SymbolicShape = [batch, 768];
      const result = resolveSymbolicShape(symbolicShape, env);

      expect(result.success).toBe(true);
      if (result.success && result.type === 'resolved') {
        expect(Array.from(result.shape)).toEqual([64, 768]);
      }
    });

    it('should validate constraints after resolution', () => {
      const env = new SymbolicEnvironment();
      const batch = env.define('batch', 32);

      // Add constraint that conflicts with binding
      env.constrain(batch, 'gt', 50); // 32 > 50 is false

      const symbolicShape: SymbolicShape = [batch, 768];
      const result = resolveSymbolicShape(symbolicShape, env);

      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.errors[0]).toContain('Constraint violation');
      }
    });

    it('should handle complex constraint chains', () => {
      const env = new SymbolicEnvironment();
      const a = env.define('a');
      const b = env.define('b');
      const c = env.define('c', 100);

      env.equal(a, b);
      env.equal(b, c);

      const symbolicShape: SymbolicShape = [a, b, c];
      const result = resolveSymbolicShape(symbolicShape, env);

      expect(result.success).toBe(true);
      if (result.success && result.type === 'resolved') {
        expect(Array.from(result.shape)).toEqual([100, 100, 100]);
      }
    });

    it('should prevent infinite loops in constraint resolution', () => {
      const env = new SymbolicEnvironment();
      const a = env.define('a');
      const b = env.define('b');

      // Create circular constraint without any concrete binding
      env.equal(a, b);
      env.equal(b, a);

      const symbolicShape: SymbolicShape = [a, b];
      const result = resolveSymbolicShape(symbolicShape, env);

      expect(result.success).toBe(false); // Should fail gracefully
    });
  });
});

// =============================================================================
// LayerShapeResolver
// =============================================================================

describe('LayerShapeResolver', () => {
  let resolver: LayerShapeResolver;

  beforeEach(() => {
    resolver = new LayerShapeResolver();
  });

  describe('Linear Layer Resolution', () => {
    it('should resolve linear layer shapes', () => {
      const env = resolver.getEnvironment();
      const batch = env.define('batch', 32);
      const inputFeatures = env.define('input_features', 784);
      const outputFeatures = 128;

      const inputShape: SymbolicShape = [batch, inputFeatures];
      const result = resolver.linear(inputShape, outputFeatures);

      expect(result.inputResolved.success).toBe(true);
      expect(result.outputShape).toEqual([batch, 128]);
    });

    it('should validate input dimensionality for linear layers', () => {
      const env = resolver.getEnvironment();
      const batch = env.define('batch', 32);

      const inputShape: SymbolicShape = [batch]; // 1D input invalid for linear layer

      expect(() => {
        resolver.linear(inputShape, 128);
      }).toThrow('Linear layer expects 2D input');
    });
  });

  describe('Attention Layer Resolution', () => {
    it('should resolve attention layer shapes', () => {
      const env = resolver.getEnvironment();
      const batch = env.define('batch', 16);
      const seqLen = env.define('seq_len', 128);
      const embedDim = env.define('embed_dim', 768);

      const inputShape: SymbolicShape = [batch, seqLen, embedDim];
      const result = resolver.attention(inputShape);

      expect(result.inputResolved.success).toBe(true);
      expect(result.outputShape).toEqual([batch, seqLen, embedDim]);
    });

    it('should validate input dimensionality for attention layers', () => {
      const env = resolver.getEnvironment();
      const batch = env.define('batch', 16);
      const seqLen = env.define('seq_len', 128);

      const inputShape: SymbolicShape = [batch, seqLen]; // 2D input invalid for attention

      expect(() => {
        resolver.attention(inputShape);
      }).toThrow('Attention layer expects 3D input');
    });
  });

  describe('Environment Access', () => {
    it('should provide access to underlying environment', () => {
      const env = resolver.getEnvironment();
      const batch = env.define('batch', 32);

      expect(env.getBind(batch)).toBe(32);
    });
  });
});

// =============================================================================
// Integration with Shape System
// =============================================================================

describe('Integration with Shape System', () => {
  describe('canBroadcastSymbolic', () => {
    it('should check broadcasting compatibility', () => {
      const env = new SymbolicEnvironment();
      const batch = env.define('batch', 32);
      env.define('features', 256);

      const shape1: SymbolicShape = [batch, 1];
      const shape2 = [1, 256]; // Concrete shape

      const result = canBroadcastSymbolic(shape1, shape2, env);

      expect(result).toBe(true);
    });

    it('should return false for unresolvable symbolic shapes', () => {
      const env = new SymbolicEnvironment();
      const batch = env.define('batch'); // Unbound

      const shape1: SymbolicShape = [batch, 256];
      const shape2 = [32, 256]; // Concrete shape

      const result = canBroadcastSymbolic(shape1, shape2, env);

      expect(result).toBe(false);
    });
  });

  describe('inferFromConcrete', () => {
    it('should infer symbolic dimension values from concrete shapes', () => {
      const env = new SymbolicEnvironment();
      const batch = env.define('batch');
      const features = env.define('features');

      const symbolicShape: SymbolicShape = [batch, features];
      const concreteShape = [32, 768];

      const result = inferFromConcrete(symbolicShape, concreteShape, env);

      expect(result.success).toBe(true);
      if (result.success && result.inferences) {
        expect(result.inferences.get('batch')).toBe(32);
        expect(result.inferences.get('features')).toBe(768);
      }
    });

    it('should validate existing bindings against concrete shapes', () => {
      const env = new SymbolicEnvironment();
      const batch = env.define('batch', 16); // Already bound
      const features = env.define('features');

      const symbolicShape: SymbolicShape = [batch, features];
      const concreteShape = [32, 768]; // Conflicts with batch binding

      const result = inferFromConcrete(symbolicShape, concreteShape, env);

      expect(result.success).toBe(false);
      if (!result.success && result.errors) {
        expect(result.errors[0]).toContain('bound to 16');
      }
    });

    it('should validate numeric dimensions', () => {
      const env = new SymbolicEnvironment();
      const batch = env.define('batch');

      const symbolicShape: SymbolicShape = [batch, 768];
      const concreteShape = [32, 512]; // Numeric dimension mismatch

      const result = inferFromConcrete(symbolicShape, concreteShape, env);

      expect(result.success).toBe(false);
      if (!result.success && result.errors) {
        expect(result.errors[0]).toContain('expected 768, got 512');
      }
    });

    it('should handle rank mismatches', () => {
      const env = new SymbolicEnvironment();
      const batch = env.define('batch');

      const symbolicShape: SymbolicShape = [batch, 768];
      const concreteShape = [32, 768, 3]; // Different rank

      const result = inferFromConcrete(symbolicShape, concreteShape, env);

      expect(result.success).toBe(false);
      if (!result.success && result.errors) {
        expect(result.errors[0]).toContain('rank mismatch');
      }
    });
  });
});

// =============================================================================
// Complex ML Scenarios
// =============================================================================

describe('Complex ML Scenarios', () => {
  describe('Transformer Architecture', () => {
    it('should handle transformer layer shapes', () => {
      const resolver = new LayerShapeResolver();
      const env = resolver.getEnvironment();
      const batch = env.define('batch', 16);
      const seqLen = env.define('seq_len', 512);
      const embedDim = env.define('embed_dim', 768);

      // Input embedding
      const inputShape: SymbolicShape = [batch, seqLen, embedDim];

      // Self-attention
      const attentionResult = resolver.attention(inputShape);
      expect(attentionResult.inputResolved.success).toBe(true);

      // Feed-forward layer after attention
      const ffInputShape: SymbolicShape = [batch, seqLen]; // 2D for linear layer
      const ffResult = resolver.linear(ffInputShape, 768);
      expect(ffResult.inputResolved.success).toBe(true);
    });
  });

  describe('CNN Architecture', () => {
    it('should handle convolutional layer shapes', () => {
      const env = new SymbolicEnvironment();
      const batch = env.define('batch', 32);
      const height = env.define('height', 224);
      const width = env.define('width', 224);
      const channels = env.define('channels', 3);

      const inputShape: SymbolicShape = [batch, height, width, channels];
      const result = resolveSymbolicShape(inputShape, env);

      expect(result.success).toBe(true);
      if (result.success && result.type === 'resolved') {
        expect(Array.from(result.shape)).toEqual([32, 224, 224, 3]);
      }
    });
  });

  describe('Dynamic Batching', () => {
    it('should handle variable batch sizes with constraints', () => {
      const env = new SymbolicEnvironment();
      const batch = env.define('batch');
      const seqLen = env.define('seq_len', 128);

      // Add constraints for valid batch sizes
      env.constrain(batch, 'gte', 1);
      env.constrain(batch, 'lte', 128);

      // Test with valid batch size
      env.bind(batch, 64);
      const inputShape: SymbolicShape = [batch, seqLen];
      const result = resolveSymbolicShape(inputShape, env);

      expect(result.success).toBe(true);
      if (result.success && result.type === 'resolved') {
        expect(Array.from(result.shape)).toEqual([64, 128]);
      }
    });

    it('should reject invalid batch sizes', () => {
      const env = new SymbolicEnvironment();
      const batch = env.define('batch');

      env.constrain(batch, 'gte', 1);
      env.bind(batch, 0); // Invalid - violates constraint

      const inputShape: SymbolicShape = [batch, 128];
      const result = resolveSymbolicShape(inputShape, env);

      expect(result.success).toBe(false);
    });
  });
});

// =============================================================================
// Error Handling and Edge Cases
// =============================================================================

describe('Error Handling and Edge Cases', () => {
  describe('Constraint Conflicts', () => {
    it('should detect contradictory numeric constraints', () => {
      expect(() => {
        const constraint = createConstraint(5, 'gt', 10);
        validateConstraint(constraint);
      }).not.toThrow(); // Validation only checks if constraint is well-formed

      // But adding it to environment should work, validation happens during resolution
      const env = new SymbolicEnvironment();
      expect(() => {
        env.constrain(5, 'gt', 10);
      }).toThrow('Invalid constraint');
    });

    it('should handle inequality constraints', () => {
      const env = new SymbolicEnvironment();
      const batch = env.define('batch');

      env.constrain(batch, 'gte', 1);
      env.constrain(batch, 'lte', 128);
      env.constrain(batch, 'ne', 13); // Superstition

      env.bind(batch, 64); // Valid
      expect(env.getBind(batch)).toBe(64);

      expect(() => {
        env.bind(batch, -1); // Violates binding validation
      }).toThrow('negative value');
    });
  });

  describe('Boundary Conditions', () => {
    it('should handle scalar shapes', () => {
      const env = new SymbolicEnvironment();
      const symbolicShape: SymbolicShape = [];
      const result = resolveSymbolicShape(symbolicShape, env);

      expect(result.success).toBe(true);
      if (result.success && result.type === 'resolved') {
        expect(Array.from(result.shape)).toEqual([]);
      }
    });

    it('should handle single dimension shapes', () => {
      const env = new SymbolicEnvironment();
      const batch = env.define('batch', 32);

      const symbolicShape: SymbolicShape = [batch];
      const result = resolveSymbolicShape(symbolicShape, env);

      expect(result.success).toBe(true);
      if (result.success && result.type === 'resolved') {
        expect(Array.from(result.shape)).toEqual([32]);
      }
    });

    it('should handle maximum rank shapes', () => {
      const env = new SymbolicEnvironment();
      const dims = Array.from({ length: 8 }, (_, i) => env.define(`dim${i.toString()}`, i + 1));

      const symbolicShape: SymbolicShape = dims;
      const result = resolveSymbolicShape(symbolicShape, env);

      expect(result.success).toBe(true);
      if (result.success && result.type === 'resolved') {
        expect(Array.from(result.shape)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
      }
    });
  });

  describe('Memory and Performance', () => {
    it('should handle large numbers of constraints efficiently', () => {
      const env = new SymbolicEnvironment();
      const batch = env.define('batch');

      // Add many non-conflicting constraints
      for (let i = 1; i <= 100; i++) {
        env.constrain(batch, 'gte', 0);
      }

      env.bind(batch, 32);
      expect(env.getBind(batch)).toBe(32);
    });

    it('should handle complex constraint systems efficiently', () => {
      // This test ensures complex constraint chains can be resolved
      const env = new SymbolicEnvironment();
      const dims = Array.from({ length: 8 }, (_, i) => env.define(`dim${i.toString()}`));

      // Create chain of equal constraints
      for (let i = 0; i < dims.length - 1; i++) {
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        env.equal(dims[i]!, dims[i + 1]!);
      }

      // Bind the first dimension to propagate through the chain
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
      env.bind(dims[0]!, 42);

      const symbolicShape: SymbolicShape = dims;
      const result = resolveSymbolicShape(symbolicShape, env);

      // Should successfully resolve the entire chain
      expect(result.success).toBe(true);
      if (result.success && result.type === 'resolved') {
        expect(Array.from(result.shape)).toEqual([42, 42, 42, 42, 42, 42, 42, 42]);
      }
    });
  });
});
