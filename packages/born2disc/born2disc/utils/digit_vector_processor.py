import math
import torch as pt


class DigitVectorProcessor:
    """
    A class to process digit vectors of a given radix.
    """
    def __init__(self,
                 *,
                 radix: int = None,
                 digits_num: int = None,
                 bit_depth: int = 64) -> None:
        self.radix = radix
        self.digits_num = digits_num
        self.bit_depth = bit_depth
        
        # Calculate the number of digits per integer and the number of integers per vector
        self.digits_per_int = math.floor((self.bit_depth - 1) / math.log2(self.radix))
        self.ints_per_vec = math.ceil(self.digits_num / self.digits_per_int)
        
        # Calculate the chunk sizes
        self.chunk_sizes = [self.digits_per_int] * (self.digits_num // self.digits_per_int)
        self.chunk_sizes = self.chunk_sizes + [self.digits_num % self.digits_per_int] * ((self.digits_num % self.digits_per_int) != 0)

        # Calculate the chunk starts and ends
        self.chunk_starts = [sum(self.chunk_sizes[:int_idx]) for int_idx in range(self.ints_per_vec)]
        self.chunk_ends = [sum(self.chunk_sizes[:int_idx+1]) for int_idx in range(self.ints_per_vec)]

        self.radix_powers = pt.tensor([self.radix ** digit_idx for digit_idx in range(self.digits_per_int)], dtype=pt.int64)

    def digit_vec2int_array(self,
                            *,
                            digit_vector: pt.Tensor = None,
                            convert_to_full_positives: bool = True) -> pt.Tensor:
        """
        Compress a digit vector to an array of integers.
        """
        assert digit_vector.ndim == 2
        assert digit_vector.shape[-1] == self.digits_num
        digit_vector = digit_vector.type(self.radix_powers.dtype)
        if convert_to_full_positives:
            digit_vector = self.convert_to_full_positives(digit_vector=digit_vector)

        int_array = pt.zeros((digit_vector.shape[0], self.ints_per_vec), dtype=pt.int64)
        for int_idx in range(self.ints_per_vec):
            int_array[:, int_idx] = pt.sum(digit_vector[:, self.chunk_starts[int_idx]:self.chunk_ends[int_idx]] * self.radix_powers[:self.chunk_sizes[int_idx]], dim=-1)

        return int_array
    
    def int_array2digit_vec(self,
                            *,
                            int_array: pt.Tensor = None,
                            convert_to_half_negatives: bool = True) -> pt.Tensor:
        """
        Decompress an array of integers to a digit vector.
        """
        assert int_array.ndim == 2
        assert int_array.shape[-1] == self.ints_per_vec

        digit_vector = pt.zeros((int_array.shape[0], self.digits_num), dtype=pt.int64)
        for int_idx in range(self.ints_per_vec):
            digit_vector[:, self.chunk_starts[int_idx]:self.chunk_ends[int_idx]] = (int_array[:, int_idx].unsqueeze(-1) // self.radix_powers[:self.chunk_sizes[int_idx]]) % self.radix

        if convert_to_half_negatives:
            digit_vector = self.convert_to_half_negatives(digit_vector=digit_vector)

        return digit_vector
    
    def convert_to_half_negatives(self,
                                  *,
                                  digit_vector: pt.Tensor) -> pt.Tensor:
        """
        Convert a digit vector to half negatives.
        """
        return pt.where(digit_vector <= self.radix // 2, digit_vector, digit_vector - self.radix)
    
    def convert_to_full_positives(self,
                                  *,
                                  digit_vector: pt.Tensor) -> pt.Tensor:
        """
        Convert a digit vector to full positives.
        """
        return pt.where(digit_vector >= 0, digit_vector, digit_vector + self.radix)
    
    def sort_int_array(self,
                       *,
                       int_array: pt.Tensor = None,
                       descending: bool = False) -> pt.Tensor:
        """
        # Sort an array of integers representing a set of digit vectors in a given radix. 
        """
        if descending:
            raise NotImplementedError
        else:
            sorted_int_array = int_array
            sort_perm = pt.arange(int_array.shape[0])
            for int_idx in range(self.ints_per_vec):
                _, cur_sort_perm = pt.sort(sorted_int_array[:, int_idx], stable=True, descending=False)
                sorted_int_array = sorted_int_array[cur_sort_perm]
                int_array = int_array[cur_sort_perm]

                neg_mask = sorted_int_array[:, int_idx] < 0
                arange = pt.arange(sorted_int_array.shape[0])
                neg_indices = arange[neg_mask]
                non_neg_indices = arange[~neg_mask]
                cur_neg_perm = pt.cat((non_neg_indices, neg_indices), dim=0)

                sorted_int_array = sorted_int_array[cur_neg_perm]
                sort_perm = sort_perm[cur_neg_perm]

        return sorted_int_array, sort_perm
    
    def sort_digit_vector(self,
                          *,
                          digit_vector: pt.Tensor = None,
                          half_negatives: bool = True,
                          descending: bool = False) -> pt.Tensor:
        """
        Sort a digit vector in a given radix.
        """
        dtype = digit_vector.dtype
        int_array = self.digit_vec2int_array(digit_vector=digit_vector, convert_to_full_positives=half_negatives)
        sorted_int_array, sort_perm = self.sort_int_array(int_array=int_array, descending=descending)
        sorted_digit_vector = self.int_array2digit_vec(int_array=sorted_int_array, convert_to_half_negatives=half_negatives)
        
        return sorted_digit_vector.type(dtype), sort_perm
    
    @staticmethod
    def two_unique2cat_unique(unq_1, unq_inv_1, unq_2, unq_inv_2):
        if len(unq_1.shape) == 1:
            unq_1 = pt.reshape(unq_1, (-1, 1))
        if len(unq_2.shape) == 1:
            unq_2 = pt.reshape(unq_2, (-1, 1))

        assert unq_inv_1.shape[0] == unq_inv_2.shape[0]
        ordinals = unq_inv_1 + unq_inv_2 * unq_1.shape[0]
        unq_ordinals, unq_inv = pt.unique(ordinals, return_inverse=True)

        merge_unq = pt.cat([unq_1[unq_ordinals % unq_1.shape[0]], unq_2[unq_ordinals // unq_1.shape[0]]], dim=-1)

        return merge_unq, unq_inv

    def unique(self, *, int_array: pt.Tensor = None) -> pt.Tensor:
        """
        Unique an array of integers representing a set of digit vectors in a given radix.
        """
        assert int_array.ndim == 2
        assert int_array.shape[-1] == self.ints_per_vec

        prev_unq, prev_unq_inv = pt.unique(int_array[..., 0], return_inverse=True)
        if self.ints_per_vec > 1:
            for int_idx in range(1, self.ints_per_vec):
                new_unq, new_unq_inv = pt.unique(int_array[..., int_idx], return_inverse=True)
                prev_unq, prev_unq_inv = self.two_unique2cat_unique(unq_1=prev_unq,
                                                                    unq_inv_1=prev_unq_inv,
                                                                    unq_2=new_unq,
                                                                    unq_inv_2=new_unq_inv)
        else:
            prev_unq = pt.reshape(prev_unq, (-1, 1))

        return prev_unq, prev_unq_inv
