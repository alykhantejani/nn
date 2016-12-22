nn.utils = {}

-- oops; someone forgot to add torch.Storage.type
-- TODO replace with torch.Storage.type when implemented
local function torch_Storage_type(self, type)
   local current = torch.typename(self)
   if not type then return current end
   if type ~= current then
      local new = torch.getmetatable(type).new()
      if self:size() > 0 then
         new:resize(self:size()):copy(self)
      end
      return new
   else
      return self
   end
end

-- tensorCache maintains a list of all tensors and storages that have been
-- converted (recursively) by calls to recursiveType() and type().
-- It caches conversions in order to preserve sharing semantics
-- i.e. if two tensors share a common storage, then type conversion
-- should preserve that.
--
-- You can preserve sharing semantics across multiple networks by
-- passing tensorCache between the calls to type, e.g.
--
-- > tensorCache = {}
-- > net1:type('torch.CudaTensor', tensorCache)
-- > net2:type('torch.CudaTensor', tensorCache)
-- > nn.utils.recursiveType(anotherTensor, 'torch.CudaTensor', tensorCache)
--
-- Implementation note: to make Lua table lookup behave correctly,
-- tensor keys are stored as actual tensor objects, while storage
-- keys are stored as the pointers themselves (as numbers).
function nn.utils.recursiveType(param, type, tensorCache)
   tensorCache = tensorCache or {}

   if torch.type(param) == 'table' then
      for k, v in pairs(param) do
         param[k] = nn.utils.recursiveType(v, type, tensorCache)
      end
   elseif torch.isTypeOf(param, 'nn.Module') or
          torch.isTypeOf(param, 'nn.Criterion') then
      param:type(type, tensorCache)
   elseif torch.isTensor(param) then
      if torch.typename(param) ~= type then
         local newparam
         if tensorCache[param] then
            newparam = tensorCache[param]
         else
            newparam = torch.Tensor():type(type)
            local storageType = type:gsub('Tensor','Storage')
            if param:storage() then
               local storage_key = torch.pointer(param:storage())
               if not tensorCache[storage_key] then
                  tensorCache[storage_key] = torch_Storage_type(
                        param:storage(), storageType)
               end
               assert(torch.type(tensorCache[storage_key]) == storageType)
               newparam:set(
                  tensorCache[storage_key],
                  param:storageOffset(),
                  param:size(),
                  param:stride()
               )
            end
            tensorCache[param] = newparam
         end
         assert(torch.type(newparam) == type)
         param = newparam
      end
   end
   return param
end

function nn.utils.recursiveResizeAs(t1,t2)
   if torch.type(t2) == 'table' then
      t1 = (torch.type(t1) == 'table') and t1 or {t1}
      for key,_ in pairs(t2) do
         t1[key], t2[key] = nn.utils.recursiveResizeAs(t1[key], t2[key])
      end
      for key,_ in pairs(t1) do
         if not t2[key] then
            t1[key] = nil
         end
      end
   elseif torch.isTensor(t2) then
      t1 = torch.isTensor(t1) and t1 or t2.new()
      t1:resizeAs(t2)
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(t1).." and "..torch.type(t2).." instead")
   end
   return t1, t2
end

function nn.utils.recursiveFill(t2, val)
   if torch.type(t2) == 'table' then
      for key,_ in pairs(t2) do
         t2[key] = nn.utils.recursiveFill(t2[key], val)
      end
   elseif torch.isTensor(t2) then
      t2:fill(val)
   else
      error("expecting tensor or table thereof. Got "
           ..torch.type(t2).." instead")
   end
   return t2
end

function nn.utils.recursiveAdd(t1, val, t2)
   if not t2 then
      assert(val, "expecting at least two arguments")
      t2 = val
      val = 1
   end
   val = val or 1
   if torch.type(t2) == 'table' then
      t1 = (torch.type(t1) == 'table') and t1 or {t1}
      for key,_ in pairs(t2) do
         t1[key], t2[key] = nn.utils.recursiveAdd(t1[key], val, t2[key])
      end
   elseif torch.isTensor(t1) and torch.isTensor(t2) then
      t1:add(val, t2)
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(t1).." and "..torch.type(t2).." instead")
   end
   return t1, t2
end

function nn.utils.recursiveCopy(t1,t2)
   if torch.type(t2) == 'table' then
      t1 = (torch.type(t1) == 'table') and t1 or {t1}
      for key,_ in pairs(t2) do
         t1[key], t2[key] = nn.utils.recursiveCopy(t1[key], t2[key])
      end
   elseif torch.isTensor(t2) then
      t1 = torch.isTensor(t1) and t1 or t2.new()
      t1:resizeAs(t2):copy(t2)
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(t1).." and "..torch.type(t2).." instead")
   end
   return t1, t2
end

function nn.utils.addSingletonDimension(...)
  local view, t, dim
  if select('#',...) < 3 then
    t, dim = select(1,...)
  else
    view, t, dim = select(1,...)
    assert(torch.isTensor(view),
           "output tensor expected, got " .. type(view))
  end

  assert(torch.isTensor(t), "input tensor expected")
  dim = dim or 1
  assert(dim > 0 and dim <= (t:dim() + 1), "invalid dimension: " .. dim
             .. '. Tensor is of ' .. t:dim() .. ' dimensions.')

  view = view or t.new()
  local size = torch.LongStorage(t:dim() + 1)
  local stride = torch.LongStorage(t:dim() + 1)

  for d = 1, dim - 1 do
    size[d] = t:size(d)
    stride[d] = t:stride(d)
  end
  size[dim] = 1
  stride[dim] = 1
  for d = dim + 1, t:dim() + 1 do
    size[d] = t:size(d - 1)
    stride[d] = t:stride(d - 1)
  end

  view:set(t:storage(), t:storageOffset(), size, stride)
  return view
end

function nn.utils.contiguousView(output, input, ...)
  output = output or input.new()
  if input:isContiguous() then
    output:view(input, ...)
  else
    output:resizeAs(input)
    output:copy(input)
    output:view(output, ...)
  end
  return output
end

-- go over specified fields and clear them. accepts
-- nn.utils.clearState(self, {'_buffer', '_buffer2'}) and
-- nn.utils.clearState(self, '_buffer', '_buffer2')
function nn.utils.clear(self, ...)
   local arg = {...}
   if #arg > 0 and type(arg[1]) == 'table' then
      arg = arg[1]
   end
   local function clear(f)
      if self[f] then
         if torch.isTensor(self[f]) then
            self[f]:set()
         elseif type(self[f]) == 'table' then
            self[f] = {}
         else
            self[f] = nil
         end
      end
   end
   for i,v in ipairs(arg) do clear(v) end
   return self
end

-- Crops the input image `input` from the center by size `cropSize`
-- @param input a 3D or 4D Tensor
-- @param a 2 element torch.LongTensor or table containing the size to crop from the
--        center of `input` along the last 2 dimensions
-- @returns a tensor of size `cropSize` which is the a view of the central `cropSize` of `input`
-- Note: the returned tensor shares the same underlying storage as `input`
function nn.utils.centerCrop(input, cropSize)
   assert(input:dim() == 3 or input:dim() == 4, "input should be a 3D or  4D tensor")
   assert(#cropSize == 2, "cropSize should have two elements only")
   local _input = input
   if input:dim() == 3 then
      _input = input:view(1, input:size(1), input:size(2), input:size(3))
   end
   assert(cropSize[1] > 0 and cropSize[1] <= _input:size(3),
         "0 < cropSize[1] <= input:size(3) not satisfied")
   assert(cropSize[2] > 0 and cropSize[2] <= _input:size(4),
        "0 < cropSize[1] <= input:size(3) not satisfied")

   local inputHeight = _input:size(3)
   local inputWidth = _input:size(4)

   local rowStart = 1 + math.floor((inputHeight - cropSize[1])/2.0)
   local rowEnd = rowStart + cropSize[1] - 1
   local colStart = 1 +  math.floor((inputWidth - cropSize[2])/2.0)
   local colEnd = colStart + cropSize[2] - 1
   if input:dim() == 3 then
      return input[{{}, {rowStart, rowEnd}, {colStart, colEnd}}]
   else
      return input[{{}, {}, {rowStart, rowEnd}, {colStart, colEnd}}]
   end
end


-- Crops the larger image from the center by the size of the smaller tensor.
-- @param t1 a 4D Tensor of shape [B,C,H,W]
-- @param t2 a 4D Tensor of shape [B,C,H,W]
-- @returns t1_crop, t2_crop - a view on each input tensor, center-cropped to the size of the smaller one
-- Note: the returned tensors shares the same underlying storage as the inputs
function nn.utils.autoCrop(t1, t2)
   assert(t1:dim() == 4 or t1:dim() == 3, "inputs should be a 3D or 4D tensors")
   assert(t1:dim() == t2:dim(), "inputs should have the same number of dimensions")
   local _t1 = t1
   local _t2 = t2
   if t1:dim() == 3 then
      _t1 = t1:view(1, t1:size(1), t1:size(2), t1:size(3))
      _t2 = t2:view(1, t2:size(1), t2:size(2), t2:size(3))
   end
   assert(_t1:size(1) == _t2:size(1), "inputs should have the same batch size")
   assert(_t1:size(2) == _t2:size(2), "inputs should have the same number of channels")
   assert((_t1:size(3) <= _t2:size(3) and _t1:size(4) <= _t2:size(4)) or
          (_t2:size(3) <= _t1:size(3) and _t2:size(4) <= _t1:size(4)),
          "One input must be smaller than or equal in size of the other")

   if (_t1:size(3) <= _t2:size(3) and _t1:size(4) <= _t2:size(4)) then
      return t1, nn.utils.centerCrop(t2, {_t1:size(3), _t1:size(4)})
   else
      return nn.utils.centerCrop(t1, {_t2:size(3), _t2:size(4)}), t2
   end
end


table.unpack = table.unpack or unpack
