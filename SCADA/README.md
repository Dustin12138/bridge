### Parse SCADA API calls into traces

Run `find_write_api_sequence.py`. It takes `ScadaBREXE.CSV` as input, which contains all API calls. It outputs two files.
- `WRITE_API_Sequences.txt`, which is the trace file. Each line starts from TCP SEND (WRITE) and ends with the next TCP SEND (WRITE), including their timestamps.
- `All_TCP_Send_and_Receive_Sequences.txt`, which contains all TCP SEND and TCP RECEIVE API calls, including their timestamps. and TCP data length. The reason we need to have this is to identify READ and WRITE commands. For the WRITE commands, the TCP data length is the same for Send and Receive.

```python
python3 find_write_api_sequence.py
```

### Timing and Frequency Analysis

Run `timing_and_frequency_analysis.py`. It takes two files, `All_TCP_Send_and_Receive_Sequences.txt` and `WRITE_API_Sequences.txt` as input, which are the outputs of `find_write_api_sequence.py`. It prints out the timing and frequency analysis on the console, including WRITE_TO_WRITE, READ_TO_WRITE, READ_TO_READ, and Cycle Frequencies (Use a WRITE TO WRITE as a cycle). It also contains their mean value and standard deviation.

```python
python3 timing_and_frequency_analysis.py
```


### Extract SCADA Dependencies API calls

Run `get_scadabr_and_dependency_api_calls.py`. It takes `All_API_Calls.CSV` as input, which contains all the API calls. It outputs `ScadaBR_Dependencies.csv`, which contains all the SCADA dependent API calls.

```python
python3 get_scadabr_and_dependency_api_calls.py
``` 

