# Enhanced Genetic Algorithm Parameter Scaling and Configuration Plan

## Problem Analysis

### Critical Issue: Hardcoded Mutation Parameters in Scaled Space
- **Current Implementation**: `sigma=0.2` in `_mutate_with_bounds()` for all parameters in [0,1] scaled space
- **Impact**: 20% mutation represents vastly different physical changes across CPPs:
  - **spray_rate**: 80-180 g/min (range: 100) → 20 g/min mutation (high sensitivity)
  - **air_flow**: 400-700 m³/h (range: 300) → 60 m³/h mutation (medium sensitivity)  
  - **carousel_speed**: 20-40 rph (range: 20) → 4 rph mutation (lower sensitivity)

### Current Limitations
1. **Parameter-agnostic mutation**: All CPPs treated equally despite different process sensitivities
2. **No configuration control**: Hardcoded mutation parameters prevent optimization tuning
3. **Suboptimal pharmaceutical control**: Fixed mutation doesn't respect process parameter importance
4. **Missing GA configurability**: Limited ability to tune for different manufacturing scenarios

## Implementation Plan

### Phase 1: Enhanced GA Configuration Structure

#### 1.1 Expand config.yaml GA section
```yaml
mpc:
  # Existing GA parameters
  population_size: 40
  generations: 15
  mutation_rate: 0.1
  crossover_rate: 0.7
  
  # NEW: Enhanced mutation configuration
  mutation_config:
    sigma: 0.1                    # Global mutation strength (reduced from 0.2)
    indpb: 0.1                    # Per-gene mutation probability
    adaptive_mutation: true       # Enable parameter-specific scaling
    
    # Parameter-specific mutation weights (pharmaceutical process expertise)
    parameter_weights:
      spray_rate: 0.5             # Lower weight - high sensitivity to particle formation
      air_flow: 1.0               # Medium weight - affects drying and flow patterns
      carousel_speed: 1.5         # Higher weight - less sensitive mechanical parameter
  
  # NEW: Advanced GA parameters
  crossover_config:
    alpha: 0.5                    # Blend crossover parameter for continuous optimization
    
  selection_config:
    tournament_size: 3            # Selection pressure (existing but now configurable)
    elite_preservation: 2         # Number of best individuals to preserve per generation
```

#### 1.2 Add pharmaceutical scenario presets
```yaml
# Scenario-specific GA configurations
ga_scenarios:
  startup_control:
    mutation_config:
      sigma: 0.15               # More exploration during startup
      parameter_weights:
        spray_rate: 0.3         # Very conservative on spray rate
        air_flow: 0.8           
        carousel_speed: 2.0     # More aggressive on mechanical parameters
        
  grade_changeover:
    mutation_config:
      sigma: 0.12               # Balanced exploration for transitions
      parameter_weights:
        spray_rate: 0.7         # Moderate spray rate changes
        air_flow: 1.2           
        carousel_speed: 1.3     
        
  disturbance_rejection:
    mutation_config:
      sigma: 0.08               # Conservative mutation for stability
      parameter_weights:
        spray_rate: 0.4         # Minimize spray rate perturbations
        air_flow: 0.9           
        carousel_speed: 1.1     
```

### Phase 2: Smart Parameter-Aware Mutation Implementation

#### 2.1 Enhance GeneticOptimizer constructor
```python
def __init__(self, param_bounds, config, fitness_function=None):
    # Existing initialization...
    
    # NEW: Process mutation configuration
    self.mutation_config = config.get('mutation_config', {})
    self.base_sigma = self.mutation_config.get('sigma', 0.1)
    self.parameter_weights = self.mutation_config.get('parameter_weights', {})
    self.adaptive_mutation = self.mutation_config.get('adaptive_mutation', False)
    
    # Calculate parameter-specific sigma values
    self._calculate_adaptive_sigmas()
```

#### 2.2 Implement adaptive mutation calculation
```python
def _calculate_adaptive_sigmas(self):
    """Calculate parameter-specific mutation strengths based on pharmaceutical expertise."""
    self.parameter_sigmas = []
    
    if not self.adaptive_mutation:
        # Use global sigma for all parameters
        self.parameter_sigmas = [self.base_sigma] * self.n_params
        return
    
    # Map parameter positions to CPP names
    cpp_names = self.config.get('cpp_names', ['spray_rate', 'air_flow', 'carousel_speed'])
    horizon = self.config['horizon']
    
    for step in range(horizon):
        for cpp_name in cpp_names:
            weight = self.parameter_weights.get(cpp_name, 1.0)
            adapted_sigma = self.base_sigma * weight
            self.parameter_sigmas.append(adapted_sigma)
```

#### 2.3 Implement parameter-aware mutation
```python
def _mutate_with_bounds(self, individual):
    """Enhanced mutation with parameter-specific sigma values."""
    if self.adaptive_mutation and len(self.parameter_sigmas) == len(individual):
        # Apply parameter-specific mutation
        for i, (sigma, bound) in enumerate(zip(self.parameter_sigmas, self.param_bounds)):
            if random.random() < self.mutation_config.get('indpb', 0.1):
                # Gaussian mutation with parameter-specific sigma
                individual[i] += random.gauss(0, sigma)
                # Enforce bounds
                low, high = bound
                individual[i] = max(low, min(high, individual[i]))
    else:
        # Fallback to standard mutation
        tools.mutGaussian(individual, mu=0, sigma=self.base_sigma, 
                         indpb=self.mutation_config.get('indpb', 0.1))
    
    self._check_bounds(individual)
    return individual,
```

### Phase 3: Configuration Integration and Validation

#### 3.1 Update RobustMPCController
```python
def _create_optimizer(self):
    """Enhanced optimizer creation with mutation configuration."""
    param_bounds = self._get_param_bounds()
    
    # Prepare enhanced GA configuration
    ga_config = self.config['ga_config'].copy() if 'ga_config' in self.config else {}
    
    # Add mutation configuration from main config
    if 'mutation_config' in self.config:
        ga_config['mutation_config'] = self.config['mutation_config']
    
    # Add CPP names for parameter mapping
    ga_config['cpp_names'] = self.config['cpp_names']
    ga_config['horizon'] = self.config['horizon']
    ga_config['num_cpps'] = len(self.config['cpp_names'])
    
    return self.optimizer_class(param_bounds, ga_config)
```

#### 3.2 Add configuration validation
```python
def _validate_mutation_config(self):
    """Validate enhanced GA mutation configuration."""
    if 'mutation_config' not in self.config:
        return  # Optional configuration
    
    mutation_config = self.config['mutation_config']
    
    # Validate sigma
    sigma = mutation_config.get('sigma', 0.1)
    if not 0.001 <= sigma <= 1.0:
        raise ValueError(f"mutation_config.sigma must be between 0.001 and 1.0, got {sigma}")
    
    # Validate parameter weights
    if 'parameter_weights' in mutation_config:
        weights = mutation_config['parameter_weights']
        for cpp_name in self.config['cpp_names']:
            if cpp_name in weights:
                weight = weights[cpp_name]
                if not 0.1 <= weight <= 5.0:
                    raise ValueError(f"Parameter weight for {cpp_name} must be between 0.1 and 5.0, got {weight}")
```

### Phase 4: Comprehensive Testing Suite

#### 4.1 Performance comparison tests
```python
def test_adaptive_mutation_performance():
    """Test that adaptive mutation improves optimization quality."""
    # Compare standard vs adaptive mutation on pharmaceutical scenarios
    # Measure convergence speed and solution quality
    pass

def test_parameter_sensitivity_scaling():
    """Validate that mutation scales appropriately respect physical parameter ranges."""
    # Test that spray_rate mutations are more conservative than carousel_speed
    pass
```

#### 4.2 Pharmaceutical scenario validation
```python
def test_pharmaceutical_scenarios():
    """Test optimization quality for different manufacturing scenarios."""
    scenarios = ['startup_control', 'grade_changeover', 'disturbance_rejection']
    # Validate that scenario-specific configurations improve performance
    pass
```

#### 4.3 Configuration compatibility tests
```python
def test_backwards_compatibility():
    """Ensure existing configurations continue to work."""
    # Test with old config format
    # Test with missing mutation_config section
    pass

def test_configuration_validation():
    """Test validation of all new configuration parameters."""
    # Test invalid sigma values
    # Test invalid parameter weights
    # Test missing CPP names in weights
    pass
```

### Phase 5: Documentation and Production Readiness

#### 5.1 Update configuration documentation
- Add detailed comments explaining mutation parameter impacts
- Provide pharmaceutical process guidance for parameter tuning
- Include scenario-specific configuration examples

#### 5.2 Performance tuning guidelines
- Recommended sigma values for different pharmaceutical processes
- Parameter weight guidelines based on process sensitivity analysis
- Optimization quality metrics and monitoring

## Expected Outcomes

### Performance Improvements
- **Better convergence**: Parameter-aware mutation improves solution quality
- **Pharmaceutical expertise**: Mutation respects process parameter sensitivities
- **Scenario optimization**: Configurable parameters for different manufacturing situations

### Production Benefits  
- **Improved control quality**: Better optimization → better pharmaceutical product quality
- **Reduced tuning time**: Preconfigured scenarios for common manufacturing situations
- **Industrial reliability**: Professional configuration management with validation

### Technical Advantages
- **Backwards compatibility**: Existing configurations continue to work
- **Extensibility**: Easy to add new parameters and pharmaceutical scenarios
- **Maintainability**: Clear separation of concerns and configuration validation

## Implementation Priority
**High Priority** - This addresses a fundamental optimization performance issue that directly impacts pharmaceutical process control quality and manufacturing outcomes.

## Files to Modify
1. `V2/config.yaml` - Enhanced GA configuration structure
2. `V2/robust_mpc/optimizers.py` - Adaptive mutation implementation  
3. `V2/robust_mpc/core.py` - Configuration integration and validation
4. `V2/tests/test_adaptive_mutation.py` - New comprehensive test suite
5. `V2/docs/` - Updated configuration documentation

## Estimated Implementation Time
- **Phase 1-2**: Configuration and core implementation (4-6 hours)
- **Phase 3**: Integration and validation (2-3 hours)  
- **Phase 4**: Comprehensive testing (3-4 hours)
- **Phase 5**: Documentation (1-2 hours)
- **Total**: 10-15 hours for complete implementation

This plan addresses the critical GA parameter scaling issue while maintaining professional standards and pharmaceutical manufacturing requirements.