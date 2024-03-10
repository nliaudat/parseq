# parseq with TensorRT Support

This repository is a fork of the original `parseq` project with modifications made to the `system.py` file to enable porting to TensorRT. The original model files had issues with porting, and the modifications in the `system.py` file address these issues, allowing for compatibility with TensorRT.

This repo has the modified `strhub/models/parseq/system.py` file, so you can either clone this repo or replace the system.py file with this one.

## Changes Made
### Change 1
In the `strhub` file, the following line:
```python
tgt_mask = query_mask = torch.triu(torch.full((num_steps, num_steps), float('-inf'), device=self._device), 1)
```
has been replaced with:
```python
tgt_mask = query_mask = torch.from_numpy(np.triu(torch.full((num_steps, num_steps), float('-inf'), device=self._device).numpy(), 1))
```

### Change 2
```python
tgt_in[:, j] = p_i.squeeze().argmax(-1)
```
has been replaced with:

```python
for one_array in tgt_in:
		one_array[j] = p_i.squeeze().argmax(-1)
```
### Change 3
```python
if self.refine_iters:
            # For iterative refinement, we always use a 'cloze' mask.
            # We can derive it from the AR forward mask by unmasking the token context to the right.
            query_mask[torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=self._device), 2)] = 0
            bos = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)
            for i in range(self.refine_iters):
                # Prior context is the previous output.
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                tgt_padding_mask = ((tgt_in == self.eos_id).int().cumsum(-1) > 0)  # mask tokens beyond the first EOS token.
                tgt_out = self.decode(tgt_in, memory, tgt_mask, tgt_padding_mask,
                                      tgt_query=pos_queries, tgt_query_mask=query_mask[:, :tgt_in.shape[1]])
                logits = self.head(tgt_out)
```
has been replaced with:

```python
if self.refine_iters:
            query_mask[torch.from_numpy(np.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=self._device).numpy(), 2))] = 0
            bos = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)
            for i in range(self.refine_iters):
                # Prior context is the previous output.
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                bool_array = (tgt_in == self.eos_id)
                final_custom_cumsum = []
                cumsum_custom = []
                for cumsum_one_array in bool_array:
                    sum =0
                    for cumsum_element in cumsum_one_array:
                        sum = sum + cumsum_element
                        cumsum_custom.append(int(sum))
                    final_custom_cumsum.append(cumsum_custom)
                final_custom_cumsum = torch.from_numpy(np.array(final_custom_cumsum))
                tgt_padding_mask = (final_custom_cumsum > 0)
                
                tgt_out = self.decode(tgt_in, memory, tgt_mask, tgt_padding_mask,
                                      tgt_query=pos_queries, tgt_query_mask=query_mask[:, :tgt_in.shape[1]])
                logits = self.head(tgt_out)
  ```

## Version used
```
pytorch : 1.13.0 
onnx oppset : 14
tensorRT : 8.5.3.1
Cuda : 11.4
```



