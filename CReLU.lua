local CReLU, parent = torch.class('nn.CReLU', 'nn.Sequential')

-- Implements the CReLU activation function as described by
-- W. Shang et al. in "Understanding and Improving Convolutional Neural Networks
-- via Concatenated Rectified Linear Units"
-- Note: Only supports batched input of the form [B,C,H,W]
function CReLU:__init(inplace)
   parent.__init(self)
   local concatTable = nn.ConcatTable()
   concatTable:add(nn.Identity())
   concatTable:add(nn.MulConstant(-1))
   self.inplace = inplace or false
   self.batched = true
   self:add(concatTable)
   self:add(nn.JoinTable(2))
   self:add(nn.ReLU(self.inplace))
end


function CReLU:updateOutput(input)
   self.batched = true
   if input:nDimension() == 3 then
      self.batched = false
   end

   if self.batched then
      parent.updateOutput(self, input)
   else
      parent.updateOutput(self, input:view(1, input:size(1), input:size(2), input:size(3)))
      self.output = self.output:view(self.output:size(2), self.output:size(3), self.output:size(4))
   end

   return self.output
end


function CReLU:updateGradInput(input, gradOutput)
   if input:nDimension() == 3 then
      self.batched = false
   end

   if self.batched then
      parent.updateGradInput(self, input, gradOutput)
   else
      parent.updateGradInput(self, input:view(1, input:size(1), input:size(2), input:size(3)),
                             gradOutput:view(1, gradOutput:size(1), gradOutput:size(2), gradOutput:size(3)))
      self.gradInput = self.gradInput:view(input:size(1), input:size(2), input:size(3))
   end

   return self.gradInput
end
