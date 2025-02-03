module Main where
import System.Environment (getArgs)
import Data.List (isPrefixOf)
import System.IO (readFile')
import Data.Char (isSpace)
main :: IO ()
main = do
  [n', file, savedir] <- getArgs
  let n = read n' :: Int
  input <- readFile' file
  let (prefix, t1s:rest) = span prefixTest $ lines input
      (compute, suffix) = span  fooEndTest rest
  writeFile (savedir ++ "/body.ins") $ fixupMultilineGroups compute
  writeFile file $ unlines $ prefix ++ [t1s, replacement n] ++ suffix
fixupMultilineGroups :: [String] -> String
fixupMultilineGroups = unlines
  -- . reverse. go []
  -- where
  --   go acc [] = acc
  --   go acc (x:xs)
  --     | "float" `isPrefixOf` trimStart x = let
  --           (block,rest) = span ('=' `elem`) xs
  --         in
  --           go (concat (x:block) : acc) rest
  --     | otherwise = go (x:acc) xs
prefixTest :: String -> Bool
prefixTest = not . ("long t1s" `isPrefixOf`) . trimStart
fooEndTest :: String -> Bool
fooEndTest = ("struct timeval t2;" /=) . trimStart
replacement :: Int -> String
replacement n = unlines $ [ "\tspmv_kernel" ++ show i ++ "(x,y,val);"| i <- [10..n+9]]
trimStart :: String -> String
trimStart = dropWhile isSpace