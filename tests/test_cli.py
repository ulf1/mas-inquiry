import pytest
from unittest.mock import patch, MagicMock
import sys
import os

from src.cli import main

@patch('src.cli.graph')
def test_cli_main(mock_graph, capsys):
    mock_event1 = {"init_node": {"status": "started"}}
    mock_event2 = {
        "some_worker": {
            "worker_replies": {
                "dim1": {"answers_list": [], "similarity_scores": [], "connections_list": []}
            }
        }
    }
    mock_event3 = {"summary_node": {"summary": "This is a summary."}}
    
    mock_graph.stream.return_value = [mock_event1, mock_event2, mock_event3]

    # Patch sys.argv
    test_args = ["cli.py", "-q", "Test query"]
    with patch.object(sys, 'argv', test_args):
        main()
        
    captured = capsys.readouterr()
    stdout = captured.out
    
    assert "Started Multi-Agent System with query: 'Test query'" in stdout
    assert "Initialization complete." in stdout
    assert "Worker (dim1) Replied:" in stdout
    assert "Final Summary:" in stdout
    assert "This is a summary." in stdout
    assert "Execution Finished" in stdout

@patch('src.cli.graph')
def test_cli_error(mock_graph, capsys):
    mock_graph.stream.side_effect = Exception("Test exception")
    
    test_args = ["cli.py", "-q", "Test fail"]
    with patch.object(sys, 'argv', test_args):
        with pytest.raises(SystemExit) as e:
            main()
            
    assert e.value.code == 1
    
    captured = capsys.readouterr()
    assert "An error occurred during execution: Test exception" in captured.out
