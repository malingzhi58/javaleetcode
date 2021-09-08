import javafx.util.Pair;

import java.io.*;
import java.util.*;
import java.text.*;
import java.math.*;
import java.util.regex.*;

public class S601 {
    public static void main(String[] args) {

    }

}


class Solution7 {

    static class TreeNode {
        String name;
        Map<String, String> attributes = new HashMap<>();
        Map<String, TreeNode> tags = new HashMap<>();
    }

    /**
     * <tag1 value = "HelloWorld">
     * <tag2 name = "Name1">
     * </tag2>
     * </tag1>
     *
     * @param tags
     * @param queries
     */
    static void queryTags(String[] tags, String[] queries) {
//        int id = 0;
//        List<TreeNode> list = new ArrayList<>();
//
//        Pair<TreeNode,Integer> t= startParse(tags,id);

    }


//    private static Pair<TreeNode, Integer> startParse(String[] tags, int id) {
//        for (; id < tags.length; ) {
//            TreeNode root = parseNode(tags[id++]);
//            TreeNode nextline= parseNode(tags[id++]);
//            if(nextline!=null){
//                String name = getName(tags[id]);
//                root.tags.put(name,nextline);
//            }
//        }
//    }


    private static String getName(String tag) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < tag.length(); i++) {
            if (tag.charAt(i) != '<' ) {
                while (tag.charAt(i) != ' ') {
                    sb.append(tag.charAt(i++));
                }
                return sb.toString();
            }
        }
        return sb.toString();
    }

    private static TreeNode parseNode(String tag) {
        if (tag.contains("/")) return null;
        TreeNode root = new TreeNode();
        StringBuilder sb = new StringBuilder();
        boolean findName = false;
        for (int i = 0; i < tag.length(); i++) {
            if (tag.charAt(i) != '<' && !findName) {
                while (tag.charAt(i) != ' ') {
                    sb.append(tag.charAt(i++));
                }
                findName = true;
                root.name = sb.toString();
                sb = new StringBuilder();
            }
            if (tag.charAt(i) != ' ') {
                while(tag.charAt(i)!=' '){
                    sb.append(tag.charAt(i++));
                }
                String valueName = sb.toString();
                sb=new StringBuilder();
                while(tag.charAt(i)!='"'){
                    i++;
                }
                i++;
                while(tag.charAt(i)!='"'){
                    sb.append(tag.charAt(i++));
                }
                root.attributes.put(valueName,sb.toString());
                sb = new StringBuilder();
            }
        }

        return root;
    }

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);

        int _tags_size = 0;
        _tags_size = Integer.parseInt(in.nextLine().trim());
        int _queries_size = 0;
        _queries_size = Integer.parseInt(in.nextLine().trim());

        String[] _tags = new String[_tags_size];
        String _tags_item;
        for (int _tags_i = 0; _tags_i < _tags_size; _tags_i++) {
            try {
                _tags_item = in.nextLine();
            } catch (Exception e) {
                _tags_item = null;
            }
            _tags[_tags_i] = _tags_item;
        }

        String[] _queries = new String[_queries_size];
        String _queries_item;
        for (int _queries_i = 0; _queries_i < _queries_size; _queries_i++) {
            try {
                _queries_item = in.nextLine();
            } catch (Exception e) {
                _queries_item = null;
            }
            _queries[_queries_i] = _queries_item;
        }

        queryTags(_tags, _queries);

    }
}
