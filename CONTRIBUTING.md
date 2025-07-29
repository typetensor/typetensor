# Contributing to TypeTensor

Thank you for your interest in contributing to TypeTensor! 🎉

TypeTensor is in early development with the API and architecture actively evolving. At this stage, we're primarily looking for **feedback and issue reports** to help shape the project's direction.

## How to Contribute Right Now

### 🐛 Report Issues

We welcome all types of feedback:

- **Bug reports** - Found something broken? Let us know!
- **API feedback** - Suggestions on the tensor operations interface
- **Documentation issues** - Unclear explanations or missing information  
- **Feature requests** - Ideas for operations or capabilities
- **Performance issues** - Slow operations or memory problems
- **Developer experience** - Setup problems or confusing workflows

### 📋 Before Opening an Issue

1. **Search existing issues** to avoid duplicates
2. **Try the latest version** - check if it's already fixed
3. **Check the examples** in [`./examples`](./examples) for usage patterns

### 🔒 Security Issues

**Never report security vulnerabilities in public issues.** Instead, email them privately to [thomas@santerre.xyz](mailto:thomas@santerre.xyz).

## Issue Reporting Guidelines

### Bug Reports

Help us fix issues faster with these details:

```markdown
**Expected Behavior**
What should happen?

**Actual Behavior** 
What actually happens?

**Steps to Reproduce**
1. 
2. 
3. 

**Environment**
- TypeScript version:
- Node.js version:
- Operating system:
- TypeTensor version:

**Code Example**
```typescript
// Minimal code that reproduces the issue
```

**Additional Context**
Any other relevant information
```

### Feature Requests

For new capabilities or improvements:

```markdown
**Problem**
What problem does this solve?

**Proposed Solution**
How should it work?

**Alternatives Considered**
Other approaches you've thought about

**Use Case**
Real-world scenario where this would help
```

## Why Only Issues for Now?

TypeTensor's architecture is actively evolving:

- **API design** - Core tensor operations interface is stabilizing
- **Type system** - Compile-time shape validation is being refined  
- **Backend interface** - Device abstraction layer is in flux
- **Build system** - Package structure and tooling are changing

We'll open up code contributions once the foundational architecture stabilizes. Your issue reports and feedback are invaluable for getting there!

## Communication

- **GitHub Issues** - Primary communication channel
- **Discussions** - Use [GitHub Discussions](https://github.com/typetensor/typetensor/discussions) for questions and ideas
- **Email** - Reach maintainer at [thomas@santerre.xyz](mailto:thomas@santerre.xyz)

## Project Goals

Understanding these helps frame good issues:

1. **Compile-time safety** - Catch tensor shape errors at build time
2. **Zero runtime overhead** - Type checking should be compile-time only
3. **Familiar API** - NumPy-like interface for JavaScript developers  
4. **Modular backends** - Support CPU, GPU, and WebAssembly execution
5. **Developer experience** - Great TypeScript integration and error messages

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). Be respectful and welcoming to all contributors.

## Recognition

All contributors will be recognized in the project. Issue reporters help shape TypeTensor's development and are valued members of the community.

---

**Questions?** Feel free to open a [discussion](https://github.com/typetensor/typetensor/discussions) or reach out via email. We're excited to hear from you! 🚀