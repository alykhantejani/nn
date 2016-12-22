--[[
   SpatialAutoCropMSECriterion.
   Implements the MSECriterion when the spatial resolution of the input is less than
   or equal to the spatial resolution of the target. It achieves this center-cropping
   the target to the same spatial resolution of the input and the MSE is then
   calculated between these cropped inputs
]]
local SpatialAutoCropMSECriterion, parent = torch.class('nn.SpatialAutoCropMSECriterion', 'nn.MSECriterion')

function SpatialAutoCropMSECriterion:__init(sizeAverage)
    parent.__init(self, sizeAverage)
end

local function inputResolutionIsSmallerThanTargetResolution(input, target)
   heightIdx = 2
   widthIdx = 3
   if input:dim() == 4 then
      heightIdx = 3
      widthIdx = 4
   end
   return input:size(heightIdx) <= target:size(heightIdx) and input:size(widthIdx) <= target:size(widthIdx)
end

function SpatialAutoCropMSECriterion:updateOutput(input, target)
   assert(input:dim() == target:dim(), "input and target should have the same number of dimensions")
   assert(input:dim() == 4 or input:dim() == 3, "input and target must have 3 or 4 dimensions")
   assert(inputResolutionIsSmallerThanTargetResolution(input, target),
   "ASSERT 1: spatial resolution of input should be less than or equal to the spatial resolution of the target")

   local inputCropped, targetCropped = nn.utils.autoCrop(input, target)
   return parent.updateOutput(self, inputCropped, targetCropped)
end


function SpatialAutoCropMSECriterion:updateGradInput(input, gradOutput)
   assert(input:dim() == gradOutput:dim(), "input and gradOutput should have the same number of dimensions")
   assert(input:dim() == 4 or input:dim() == 3, "input and gradOutput must have 3 or 4 dimensions")
   assert(inputResolutionIsSmallerThanTargetResolution(input, gradOutput),
   "spatial resolution of input should be less than or equal to the spatial resolution of the gradOutput")

   local inputCropped, gradOutputCropped = nn.utils.autoCrop(input, gradOutput)
   return parent.updateGradInput(self, inputCropped, gradOutputCropped)
end
