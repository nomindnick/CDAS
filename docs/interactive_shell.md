# CDAS Interactive Shell

The CDAS Interactive Shell provides a user-friendly command-line interface for working with the Construction Document Analysis System. It offers features like command history, tab completion, and contextual help to make working with construction documents more efficient.

## Getting Started

To start the interactive shell, run:

```bash
python -m cdas.cli shell
```

Or if you have the CDAS package installed:

```bash
cdas shell
```

## Shell Features

The interactive shell offers several advantages over the traditional command-line interface:

- **Command History**: Previous commands are stored and can be accessed using the up and down arrow keys.
- **Tab Completion**: Press Tab to complete commands, file paths, document IDs, and more.
- **Contextual Help**: Get detailed help on commands and their arguments.
- **Colored Output**: Visual distinction between different types of information (if terminal supports colors).
- **Context Management**: Set project context to work within a specific project.
- **Tutorials and Examples**: Built-in tutorials and examples help new users get started.

## Command Structure

The shell maintains the same command structure as the CLI, but in a more interactive format:

```
cdas> COMMAND [SUBCOMMAND] [ARGUMENTS] [OPTIONS]
```

### Basic Commands

- `ingest` - Process and ingest a document into the system
- `list` - List documents in the system
- `show` - Show details of a specific document
- `patterns` - Detect financial patterns
- `amount` - Analyze a specific amount
- `analyze` - Analyze a document
- `search` - Search for text in documents
- `find` - Find line items by amount range
- `ask` - Ask a natural language question
- `report` - Generate various types of reports

### Help Commands

- `help` or `?` - List all available commands
- `help COMMAND` - Get detailed help on a specific command
- `tutorial` - Show a tutorial for CDAS with examples
- `examples` - Show examples of common CDAS commands

### Context Management

- `context` - Show or clear the current context
- `project` - Set the current project

### Shell Management

- `exit` or `quit` - Exit the shell
- Ctrl+D - Exit the shell

## Tab Completion

The shell offers context-aware tab completion for:

- Command names
- Document IDs
- Document types (e.g., invoice, change_order)
- Party types (e.g., contractor, district)
- Project IDs
- File paths
- Command options and flags

## Example Session

```
$ python -m cdas.cli shell

Construction Document Analysis System (CDAS) - Interactive Shell
----------------------------------------------------------------
Type 'help' or '?' to list commands.
Type 'help <command>' for detailed help on a specific command.
Type 'quit' or 'exit' to exit.

Common commands:
  ingest - Process and ingest a document into the system
  list   - List documents in the system
  show   - Show details of a specific document
  search - Search for text in documents
  ask    - Ask a natural language question about the data
  report - Generate various types of reports

cdas> ingest contract.pdf --type contract --party district
Document: contract.pdf
Type: contract
Party: district
Line items: 15

cdas> list --type contract
Found 1 documents:
ID       | Type    | Party    | Date       | File Name
----------------------------------------------
doc_123a | contract | district | 2023-05-15 | contract.pdf

cdas> show doc_123a
Document ID: doc_123a
File: contract.pdf
Type: contract
Party: district
Date created: 2023-05-15

cdas> project school_123
Project context set to: school_123

cdas:school_123> exit
Exiting CDAS shell...
```

## Getting Help

To get comprehensive help within the shell, use:

```
cdas> tutorial
```

This will display a general tutorial with a list of available topics. For specific tutorial topics:

```
cdas> tutorial documents
```

To see examples of specific commands:

```
cdas> examples report
```

## Command History

Command history is saved to `~/.cdas_history` and is preserved between sessions. Use the up and down arrow keys to navigate through previous commands.