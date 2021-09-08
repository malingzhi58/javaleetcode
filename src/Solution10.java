import java.io.*;
import java.nio.*;
import java.util.*;
import java.text.*;
import java.math.*;
public class Solution10 {
    public static class WAL {
        /**
         * Construct the WAL
         *
         * @param input the raw binary WAL
         */
        ArrayList<String> res = new ArrayList<>();

        public WAL(byte[] input) {
            // TODO
            int id = 0;
            while (id < input.length) {
                byte[] time = Arrays.copyOfRange(input, id, id + 8);
                ByteBuffer bb = ByteBuffer.wrap(time);
                long l = bb.getLong();
                id+=8;
//                System.out.println(l);
                byte mid =input[id];
                int type = mid & 0xFF;
                id++;
//                System.out.println(type);
                String msgName="";
                if(type==0){
                    msgName="INSERT";
                    short s = (short) ((input[id] & 0xFF) << 8 | (input[id+1] & 0xFF));
                    id+=2;
                    byte[] keybyte = Arrays.copyOfRange(input, id, id + s);
                    String key = new String(keybyte);
                    id+=s;
                    short s2 = (short) ((input[id] & 0xFF) << 8 | (input[id+1] & 0xFF));
                    id+=2;
                    byte[] valbyte = Arrays.copyOfRange(input, id, id + s2);
                    String val = new String(valbyte);
                    id+=s2;
                    String result =l+"|"+msgName+"|"+key+"|"+val;
                    res.add(result);
                }
                if(type==1){
                    msgName="UPSERT";
                    short s = (short) ((input[id] & 0xFF) << 8 | (input[id+1] & 0xFF));
                    id+=2;
                    byte[] keybyte = Arrays.copyOfRange(input, id, id + s);
                    String key = new String(keybyte);
                    id+=s;
                    short s2 = (short) ((input[id] & 0xFF) << 8 | (input[id+1] & 0xFF));
                    id+=2;
                    byte[] valbyte = Arrays.copyOfRange(input, id, id + s2);
                    String val = new String(valbyte);
                    id+=s2;
                    String result =l+"|"+msgName+"|"+key+"|"+val;
                    res.add(result);
                }
                if(type==2){
                    msgName="DELETE";
                    short s = (short) ((input[id] & 0xFF) << 8 | (input[id+1] & 0xFF));
                    id+=2;
                    byte[] keybyte = Arrays.copyOfRange(input, id, id + s);
                    String key = new String(keybyte);
                    id+=s;
//                    short s2 = (short) ((input[id] & 0xFF) << 8 | (input[id+1] & 0xFF));
//                    id+=2;
//                    byte[] valbyte = Arrays.copyOfRange(input, id, id + s2);
//                    String val = new String(valbyte);
//                    id+=s2;
                    String result =l+"|"+msgName+"|"+key;
                    res.add(result);
                }

            }

        }

        /**
         * Retrieve all events contained within the WAL as their string values in time order
         * DML Event String Format: "<epoch_milli>|<message_name>|<key>|<value>"
         *
         * @return a time-ordered sequence of DML Event strings
         */
        public ArrayList<String> getEvents() {
            // TODO
            return res;
        }
    }

    public static class InputParser {
        private byte[] hex2bytes(String hex) {
            int len = hex.length();
            byte[] data = new byte[len / 2];
            for (int i = 0; i < len; i += 2) {
                data[i / 2] = (byte) ((Character.digit(hex.charAt(i), 16) << 4)
                        + Character.digit(hex.charAt(i+1), 16));
            }
            return data;
        }

        public byte[] getStdin() {
            Scanner scanner = new Scanner(System.in);
            return hex2bytes(scanner.next().trim());
        }
    }

    public static void main(String args[] ) throws Exception {
        byte[] input = new InputParser().getStdin();
        WAL wal = new WAL(input);
        System.out.println("Events:");
        System.out.println(String.join("\n", wal.getEvents()));
    }
}