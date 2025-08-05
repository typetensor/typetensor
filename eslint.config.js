import eslint from '@eslint/js';
import tseslint from 'typescript-eslint';
import prettierConfig from 'eslint-config-prettier';
import prettierPlugin from 'eslint-plugin-prettier';

export default tseslint.config(
  eslint.configs.recommended,
  ...tseslint.configs.strictTypeChecked,
  ...tseslint.configs.stylisticTypeChecked,
  prettierConfig,
  {
    files: ['**/*.ts', '**/*.tsx'],
    languageOptions: {
      parser: tseslint.parser,
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
        ecmaVersion: 'latest',
        sourceType: 'module',
      },
    },
    plugins: {
      '@typescript-eslint': tseslint.plugin,
      prettier: prettierPlugin,
    },
    rules: {
      // TypeScript strict rules
      '@typescript-eslint/no-explicit-any': 'error',
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
      '@typescript-eslint/explicit-function-return-type': ['error', {
        allowExpressions: true,
        allowTypedFunctionExpressions: true,
      }],
      '@typescript-eslint/no-non-null-assertion': 'error',
      '@typescript-eslint/strict-boolean-expressions': 'error',
      '@typescript-eslint/no-floating-promises': 'error',
      '@typescript-eslint/no-misused-promises': 'error',
      '@typescript-eslint/await-thenable': 'error',
      '@typescript-eslint/no-unnecessary-type-assertion': 'error',
      '@typescript-eslint/prefer-nullish-coalescing': 'error',
      '@typescript-eslint/prefer-optional-chain': 'error',
      '@typescript-eslint/prefer-string-starts-ends-with': 'error',
      '@typescript-eslint/prefer-includes': 'error',
      '@typescript-eslint/prefer-reduce-type-parameter': 'error',
      '@typescript-eslint/no-inferrable-types': 'error',
      '@typescript-eslint/consistent-type-imports': ['error', {
        prefer: 'type-imports',
        fixStyle: 'inline-type-imports',
      }],
      '@typescript-eslint/consistent-type-exports': 'error',
      
      // General best practices
      'no-console': ['warn', { allow: ['warn', 'error'] }],
      'no-debugger': 'error',
      'no-alert': 'error',
      'prefer-const': 'error',
      'no-var': 'error',
      'eqeqeq': ['error', 'always'],
      'curly': ['error', 'all'],
      'no-throw-literal': 'error',
      
      // Prettier integration
      'prettier/prettier': ['error', {
        semi: true,
        trailingComma: 'all',
        singleQuote: true,
        printWidth: 100,
        tabWidth: 2,
        useTabs: false,
      }],

      // Certain rules that are not really useful for our ergonomics
      // no need to bother with require-await, many of the storage interfaces have inconsistent sync/async methods
      '@typescript-eslint/require-await': 'off',
      // Allow deprecated methods from expect-type
      '@typescript-eslint/no-deprecated': 'off',
      // Being verbose with types and unnecessary types is useful for our library and readability
      '@typescript-eslint/no-redundant-type-constituents': 'off',
      '@typescript-eslint/no-unnecessary-type-parameters': 'off',
      '@typescript-eslint/no-unnecessary-type-arguments': 'off',
      '@typescript-eslint/no-unnecessary-type-assertion': 'off',
      '@typescript-eslint/restrict-template-expressions': 'off',
      '@typescript-eslint/explicit-function-return-type': 'off',
    },
  },
  {
    files: ['**/*.js'],
    ...tseslint.configs.disableTypeChecked,
  },
  {
    files: ['examples/**/*.ts'],
    rules: {
      'no-console': 'off', // Allow console.log in examples
    },
  },
  {
    ignores: ['**/node_modules/', '**/dist/', '**/coverage/', '*.config.js', '*.test-d.ts', '*.test.ts', '**/*.test-d.ts', '**/*.test.ts', 'examples/**/*.ts'],
  }
);